# 2 机 8 卡跨机 NVLink 测试（arm64 + L20A / CUDA 13.0）

记录在两台通过跨机 NVLink（MNNVL）互联的 arm64 机器（各 4x NVIDIA L20A）上
跑通 `tests/elastic/test_ep.py` 的步骤与结果。单机环境搭建见
[container_build_and_test.md](container_build_and_test.md)。

## 1. 环境

- node0: `10.87.79.103`，node1: `10.87.79.102`，各 4x L20A
- 两机 GPU 处于同一 NVLink fabric 域（`nvidia-smi -q` 中 Fabric State: Completed），
  镜像默认 `NCCL_MNNVL_ENABLE=1`，8 卡组成单一 NVLink 域
- 两台机器均按 container_build_and_test.md 的命令启动容器（关键是
  `-e NVIDIA_VISIBLE_DEVICES=all`，否则驱动不会注入容器）

## 2. 第二台机器准备

在 node0 上把编译好的源码目录同步过去（wheel 已在 `dist/` 里，无需重编）：

```bash
rsync -a /home/lyh437841/DeepEP lyh437841@10.87.79.102:/home/lyh437841/
```

在 node1 容器内安装：

```bash
pip install /home/lyh437841/DeepEP/dist/deep_ep-2.1.0+local-cp312-cp312-linux_aarch64.whl
```

## 3. 运行 8 卡测试

`init_dist()`（`deep_ep/utils/envs.py`）通过 `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK`
环境变量支持多机，`WORLD_SIZE` 为节点数、`RANK` 为节点序号。

注意：本环境宿主机网卡为 `eth0`，需覆盖镜像内默认的 `bond1` 相关变量。

node0（10.87.79.103）容器内：

```bash
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
       NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export MASTER_ADDR=10.87.79.103 MASTER_PORT=8361 WORLD_SIZE=2 RANK=0
cd /tmp && python /home/lyh437841/DeepEP/tests/elastic/test_ep.py --num-processes 4
```

node1（10.87.79.102）容器内：同上，仅 `RANK=1`。

## 4. 测试结果（2026-07-21）

144 个用例组合全部通过，两节点日志均无报错，退出码 0。
所有流量走 NVLink（SO 恒为 0 GB/s，即无 scaleout/RDMA 流量，
证明跨机走的是 MNNVL 单域）。

参考性能（8 卡跨机，SU 即 NVLink 域内带宽）：
dispatch ~670-690 GB/s，combine ~730 GB/s。

完整日志：[logs/test_ep_8gpu_node0.log](logs/test_ep_8gpu_node0.log)、
[logs/test_ep_8gpu_node1.log](logs/test_ep_8gpu_node1.log)。

## 5. 跨机改走 RDMA（节点内仍走 NVLink）

DeepEP elastic 的域划分来自 NCCL 的 LSA 域（`csrc/kernels/backend/nccl.cu`
`get_physical_domain_size()`）。设 `NCCL_MNNVL_ENABLE=0` 可禁用跨机 NVLink，
NVLink 域收缩为单机 4 卡，跨机流量自动改走 RDMA（RoCE，`mlx5_bond_*`），
日志中 SO 带宽将变为非零：

```bash
export NCCL_MNNVL_ENABLE=0        # 关键开关
# 以下镜像内已预设：NCCL_IB_HCA=mlx5_bond、NCCL_IB_GID_INDEX=3（RoCE v2）
```

测试命令不变（`--allow-hybrid-mode` 默认为 1，即 RDMA+NVLink 混合模式）。
