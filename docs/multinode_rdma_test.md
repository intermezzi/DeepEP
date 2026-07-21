# 2 机 8 卡跨机 RDMA 测试（未通过，问题记录）

在 [multinode_nvlink_test.md](multinode_nvlink_test.md) 同一环境（2x arm64 节点，
各 4x L20A，RoCE LAG `mlx5_bond_0~3`，ACCL-N NCCL 2.30.4.19）上，
尝试跨机走 RDMA、节点内走 NVLink 的混合模式。**当前未跑通**，本文记录复现命令、
实际报错与已排除项。

## 1. 复现命令

在 NVLink 测试的基础上只需增加 2 个环境变量：

- `NCCL_MNNVL_ENABLE=0`：禁用跨机 NVLink，NVLink 域收缩为单机 4 卡，
  跨机流量改走 RDMA（生效后日志 Config 显示 `Ranks: 2 x 4`，NVLink 全域时为 `1 x 8`）
- `EP_NIC_NAME=mlx5_bond_0`：本环境 IB 设备名为 `mlx5_bond_*`，
  不设则 `get_rdma_gbs()` 找不到默认的 `mlx5_0` 返回 0，
  `get_theoretical_num_sms()` 除零（`ZeroDivisionError`）

node0（10.87.79.103）容器内：

```bash
export NCCL_MNNVL_ENABLE=0 EP_NIC_NAME=mlx5_bond_0
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
       NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export MASTER_ADDR=10.87.79.103 MASTER_PORT=8362 WORLD_SIZE=2 RANK=0
cd /tmp && python /home/lyh437841/DeepEP/tests/elastic/test_ep.py \
    --num-processes 4 --test-first-only
```

node1（10.87.79.102）容器内：同上，仅 `RANK=1`。

## 2. 实际报错（2026-07-21）

完整日志：[logs/test_ep_8gpu_rdma_node0.log](logs/test_ep_8gpu_rdma_node0.log)、
[logs/test_ep_8gpu_rdma_node1.log](logs/test_ep_8gpu_rdma_node1.log)。

第一个 dispatch 用例即失败。node0 侧 GPU kernel 等跨机数据超时后 trap：

```text
DeepEP hybrid dispatch (forwarding) timeout, scale-out: 0, scale-up: 2, channel: 32, lane: 0, old scale-out tail: 0, scale-out tail: (1, 16)
DeepEP NVLink barrier timeout, tag: 7, nvl: 0, thread: 0, status: 1, signal: 1, phase: 1, target: 4, counter: 2
...
RuntimeError: CUDA driver exception (csrc/jit/handle.hpp:86): 719 (CUDA_ERROR_LAUNCH_FAILED, unspecified launch failure)
```

node1 侧 CPU 等到的跨机接收计数全为 0（即对端 RDMA 写完全没有到达）：

```text
RuntimeError: Dispatch CPU wait exception (csrc/elastic/buffer.hpp:1063): CPU side received count (scaleup: 1): 0 0 0 0 # 0 0 ...
```

宿主机 dmesg 仅有 kernel trap 的后果（`NVRM: Xid 43`），无 mlx5/CQE 网卡错误。

## 3. 已排除项

- **控制面正常**：`NCCL_DEBUG=INFO,SUBSYS=NET` 显示 GIN 的 main/companion QP
  全部建连成功（含跨机 remote_rank），GDRDMA/dmabuf 注册正常
- **驱动前置条件满足**：`PeerMappingOverride=1`、`nvidia_peermem` 已加载
- **`NCCL_GIN_TYPE=PROXY`**：此版本 PROXY 不支持 railed 模式
  （`railedGinType == NCCL_GIN_TYPE_NONE`），DeepEP hybrid 模式在
  `csrc/kernels/backend/nccl.cu:88` 直接 assert，走不通
- **`NCCL_GIN_GDAKI_NIC_HANDLER=1`（CPU 敲 doorbell）**：同样的 719 超时，
  说明问题不在 doorbell 写入方式

## 4. 结论与后续

控制面（QP 建连、内存注册）全通，但 GPU 发起的跨机 RDMA 写（GDAKI/IBGDA
数据面）完全无法到达对端。疑似该 RoCE LAG bond（nexus_ib 栈）+ L20A 平台
对 GPU-initiated RDMA 的支持问题，需要与平台 / ACCL 团队确认。

后续可用 `tests/elastic/test_barrier.py` 等最小用例进一步隔离验证
GIN RDMA 写数据面是否可用。
