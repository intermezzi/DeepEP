#pragma once

// GIN / QP-level diagnostics for hang debugging.
//
// The hybrid dispatch data-plane issues GPU-initiated RDMA (GDA-KI / DOCA
// GPUNetIO) writes as fire-and-forget: `put`/`put_value`/`signal` post a WQE and
// ring the doorbell, but the send completion queue is never polled on the fast
// path (only under NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG). So when a rail QP wedges
// (link flap, QP transition to ERROR, retry/RNR exhausted, remote access fault),
// the peer simply never receives our writes and the kernel spins in an
// application-level `timeout_while` with no idea *why*.
//
// This header lets a hang site crack open the GDA-KI context, walk the send QP(s)
// toward the relevant scale-out peer, and print the real QP/CQ state plus any
// completion-with-error syndrome the NIC wrote. That turns a bare "forwarding
// timeout" into an actionable "TRANSPORT_RETRY_EXC_ERR on qpn 0x… toward peer …".
//
// Build requirement: the NCCL install must ship the GDA-KI / DOCA GPUNetIO device
// headers (it does when the IB GDA-KI backend is built). They live next to
// gin_gdaki.h under the NCCL include root (see nccl `src/Makefile`,
// `DOCA_INC_INSTALL := $(INCDIR)/nccl_device/gin/gdaki/doca_gpunetio`):
//     <nccl_root>/include/nccl_device/gin/gdaki/doca_gpunetio/doca_gpunetio_device.h
// `<nccl_root>/include` is already on DeepEP's include path. If a build does not
// have these headers, compile with -DEP_ENABLE_GIN_QP_DEBUG=0 to stub it out.

#ifndef EP_ENABLE_GIN_QP_DEBUG
#define EP_ENABLE_GIN_QP_DEBUG 1
#endif

#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/ptx.cuh>

#if EP_ENABLE_GIN_QP_DEBUG

#include <cstdint>
#include <cstdio>

// gin_gdaki.h sets these before pulling in DOCA; mirror it in case this header is
// the first to include the DOCA device code in a translation unit.
#ifndef DOCA_VERBS_USE_CUDA_WRAPPER
#define DOCA_VERBS_USE_CUDA_WRAPPER
#endif
#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif
#include <nccl_device/gin/gdaki/gin_gdaki_device_host_common.h>          // ncclGinGdakiGPUContext
#include <nccl_device/gin/gdaki/doca_gpunetio/doca_gpunetio_device.h>    // doca_gpu_dev_verbs_{qp,cq,cqe}

namespace deep_ep::elastic::comm::debug {

template <typename T>
__device__ __forceinline__ T ld_vol(const T* p) {
    return *reinterpret_cast<const volatile T*>(p);
}

// The `syndrome` byte of an mlx5 completion-with-error CQE. These are the codes to
// grep for: the *_RETRY_EXC / RNR ones mean "peer/link unreachable", FLUSH means
// "this WQE was flushed because the QP was already in ERROR" (i.e. a *later* WQE
// on the same QP is the real culprit), and the *ACCESS/PROT ones mean a memory
// key/permission problem rather than a fabric fault.
__device__ __forceinline__ const char* mlx5_cqe_syndrome_str(uint8_t s) {
    switch (s) {
        case 0x01: return "LOCAL_LENGTH_ERR";
        case 0x02: return "LOCAL_QP_OP_ERR";
        case 0x04: return "LOCAL_PROT_ERR";
        case 0x05: return "WR_FLUSH_ERR";
        case 0x06: return "MW_BIND_ERR";
        case 0x10: return "BAD_RESP_ERR";
        case 0x11: return "LOCAL_ACCESS_ERR";
        case 0x12: return "REMOTE_INVAL_REQ_ERR";
        case 0x13: return "REMOTE_ACCESS_ERR";
        case 0x14: return "REMOTE_OP_ERR";
        case 0x15: return "TRANSPORT_RETRY_EXC_ERR";
        case 0x16: return "RNR_RETRY_EXC_ERR";
        case 0x22: return "REMOTE_ABORTED_ERR";
        default:   return "UNKNOWN";
    }
}

struct cqe_err_t {
    bool found;
    uint32_t slot;          // ring slot the error CQE sits in
    uint8_t opcode;         // 13 = REQ_ERR, 14 = RESP_ERR
    uint8_t syndrome;       // mlx5 CQE syndrome (see mlx5_cqe_syndrome_str)
    uint8_t vendor;         // vendor_err_synd
    uint8_t hw_synd;        // hw_err_synd
    uint8_t hw_synd_type;   // hw_synd_type
    uint16_t wqe_counter;   // WQE index that failed
    uint32_t total;         // total error CQEs seen in the ring
};

// Scan a send CQ ring for any completion-with-error. The fast path never advances
// cqe_ci nor requests per-WQE CQEs, but the NIC still writes an error CQE for the
// failing WQE (and flush-error CQEs for the WQEs behind it) when a QP goes to
// ERROR, so a ring scan reliably surfaces the fault. Owner-bit/wqe_counter
// validation is intentionally skipped: on a hang we would rather over-report a
// stale CQE than miss the one that explains the stall.
__device__ __forceinline__ cqe_err_t scan_send_cq_errors(struct doca_gpu_dev_verbs_cq* cq) {
    cqe_err_t out{};
    auto* base = reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<uintptr_t*>(&cq->cqe_daddr)));
    if (base == nullptr)
        return out;

    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    const uint32_t scan = cqe_num < 4096u ? cqe_num : 4096u;
    for (uint32_t i = 0; i < scan; ++ i) {
        auto* cqe = reinterpret_cast<struct doca_gpunetio_ib_mlx5_cqe64*>(
            base + static_cast<uint64_t>(i) * DOCA_GPUNETIO_VERBS_CQE_SIZE);
        const uint8_t op_own = ld_vol(&cqe->op_own);
        const uint8_t opcode = op_own >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
        if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR or
            opcode == DOCA_GPUNETIO_IB_MLX5_CQE_RESP_ERR) {
            if (not out.found) {
                auto* e = reinterpret_cast<struct doca_gpunetio_ib_mlx5_err_cqe_ex*>(cqe);
                out.found = true;
                out.slot = i;
                out.opcode = opcode;
                out.syndrome = e->syndrome;
                out.vendor = e->vendor_err_synd;
                out.hw_synd = e->hw_err_synd;
                out.hw_synd_type = e->hw_synd_type;
                out.wqe_counter = doca_gpu_dev_verbs_bswap16(e->wqe_counter);
            }
            out.total += 1;
        }
    }
    return out;
}

// Dump local send-QP state (and any error CQE) for every GIN context that has
// posted work toward `dst_scaleout_rank` on the rail team. Because the rail is a
// connected RC pair, a rail fault surfaces as an error CQE on *our* send QP to the
// peer too, so the caller (a stalled forward warp) can observe the root cause
// locally. Call from a single thread (e.g. behind `ptx::elect_one_sync()`).
__device__ __forceinline__ void dump_scaleout_qp_state(
    const handle::NCCLGin& gin,
    const int dst_scaleout_rank,
    const int scaleout_rank_idx, const int scaleup_rank_idx, const int channel_idx,
    const char* where) {
    const auto& comm = gin.nccl_dev_comm;

    // Same (team, rank) -> QP-array-index translation `gin.put()` uses internally,
    // reusing the rail team the wrapper already built.
    const int peer = nccl::gin::internal::teamRankToGinRank(comm, gin.team_rail, dst_scaleout_rank);

    const int num_ctx  = static_cast<int>(comm.ginContextCount);
    const int num_conn = static_cast<int>(comm.ginConnectionCount);

    bool printed_any = false;
    // Mirror NCCL's own context iteration (cf. the QP flush loop in comm.cuh): a
    // global context index i splits into (connectionId = i % nConn, contextId =
    // i / nConn), and `ginHandles[connectionId]` is that connection's per-context
    // GDA-KI array base. Sweeping all of them visits every QP the kernel could
    // have used to reach this peer, across every GIN connection; connections whose
    // backend is not GDA-KI/DOCA are skipped (their handle is a different layout).
    for (int i = 0; i < num_ctx; ++ i) {
        const int conn = num_conn > 0 ? i % num_conn : 0;
        const int ctx  = num_conn > 0 ? i / num_conn : i;
        if (comm.ginNetDeviceTypes[conn] != NCCL_NET_DEVICE_GIN_GDAKI)
            continue;
        auto* ctx_base = static_cast<struct ncclGinGdakiGPUContext*>(comm.ginHandles[conn]);
        if (ctx_base == nullptr)
            continue;

        struct doca_gpu_dev_verbs_qp* qp = ctx_base[ctx].gdqp + peer;
        const uint64_t sq_posted = ld_vol(&qp->sq_rsvd_index);   // WQE slots SW reserved/posted
        const uint64_t sq_pi     = ld_vol(&qp->sq_wqe_pi);       // producer index (doorbell)
        const uint64_t cq_ci     = ld_vol(&qp->cq_sq.cqe_ci);    // completions SW consumed
        const uint32_t qpn       = __ldg(&qp->sq_num);           // QP number (matches host/switch logs)
        const uint32_t cqn       = __ldg(&qp->cq_sq.cq_num);
        const cqe_err_t err = scan_send_cq_errors(&qp->cq_sq);

        // A context that never sent to this peer and has a clean CQ is noise.
        if (sq_posted == 0 and not err.found)
            continue;
        printed_any = true;

        // NOTE: sq_posted >> cq_ci is EXPECTED here (fast path never polls the CQ),
        // so it is not itself a fault signal -- an error CQE is the smoking gun.
        if (err.found) {
            printf("DeepEP GIN QP [%s] ERROR: so %d su %d ch %d -> peer_so %d | conn %d ctx %d qpn %#x cqn %#x | "
                   "sq_posted %llu sq_pi %llu cq_ci %llu | CQE#%u op %u syndrome %#x(%s) "
                   "vendor %#x hw_synd %#x hw_synd_type %#x wqe_cnt %u | err_cqes %u\n",
                   where, scaleout_rank_idx, scaleup_rank_idx, channel_idx, dst_scaleout_rank,
                   conn, ctx, qpn, cqn,
                   static_cast<unsigned long long>(sq_posted),
                   static_cast<unsigned long long>(sq_pi),
                   static_cast<unsigned long long>(cq_ci),
                   err.slot, err.opcode, err.syndrome, mlx5_cqe_syndrome_str(err.syndrome),
                   err.vendor, err.hw_synd, err.hw_synd_type, err.wqe_counter, err.total);
        } else {
            printf("DeepEP GIN QP [%s] ok: so %d su %d ch %d -> peer_so %d | conn %d ctx %d qpn %#x cqn %#x | "
                   "sq_posted %llu sq_pi %llu cq_ci %llu | no error CQE (in-flight %lld)\n",
                   where, scaleout_rank_idx, scaleup_rank_idx, channel_idx, dst_scaleout_rank,
                   conn, ctx, qpn, cqn,
                   static_cast<unsigned long long>(sq_posted),
                   static_cast<unsigned long long>(sq_pi),
                   static_cast<unsigned long long>(cq_ci),
                   static_cast<long long>(sq_posted) - static_cast<long long>(cq_ci));
        }
    }
    if (not printed_any) {
        printf("DeepEP GIN QP [%s]: so %d su %d ch %d -> peer_so %d | no posted work across %d ctx "
               "(our send side is idle -- peer likely never got a chance to send us data)\n",
               where, scaleout_rank_idx, scaleup_rank_idx, channel_idx, dst_scaleout_rank, num_ctx);
    }
}

}  // namespace deep_ep::elastic::comm::debug

#else  // EP_ENABLE_GIN_QP_DEBUG

namespace deep_ep::elastic::comm::debug {
__device__ __forceinline__ void dump_scaleout_qp_state(
    const handle::NCCLGin&, int, int, int, int, const char*) {}
}  // namespace deep_ep::elastic::comm::debug

#endif  // EP_ENABLE_GIN_QP_DEBUG
