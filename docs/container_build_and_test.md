# 容器内编译与测试指南（arm64 + Blackwell / CUDA 13.0）

记录在 PAI arm64 加速镜像容器内编译 DeepEP 并跑通 `tests/elastic/test_ep.py` 的完整步骤。

## 1. 启动容器

```bash
sudo pouch run -d \
  --privileged \
  --name lyh_acclep_test \
  -v /home/lyh437841:/home/lyh437841 \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
  --shm-size=500g \
  --net=host \
  dsw-registry.ap-southeast-1.cr.aliyuncs.com/pai/training:26.06.01-pytorch2.11-python3.12-cuda13.0-ubuntu24.04-arm64_accelerated \
  tail -f /dev/null

sudo pouch exec -it lyh_acclep_test bash
```

镜像环境：Python 3.12 / PyTorch 2.11 / CUDA 13.0 (nvcc V13.0.88) / Ubuntu 24.04 arm64。

## 2. 获取源码

基于 `features/r2.0-ep128-mutli-signal` 分支的 `a291cebd` 提交（本仓库
`features/r2.0-ep128-debug` 分支即该基线加上下述两个修改）。

```bash
cd /root
git clone -b features/r2.0-ep128-debug <本仓库地址> DeepEP
```

## 3. 本分支包含的两个关键修改

1. **`deep_ep/utils/envs.py`**：RDMA 带宽检测适配 RoCE LAG/bond 和容器环境，
   新增 `EP_NIC_NAME` 环境变量覆盖网卡名，`get_rdma_gbs()` 依次尝试
   bond 聚合速率 → ibstat → sysfs port rate → netdev speed。
2. **`deep_ep/include/deep_ep/common/ptx.cuh`**：cherry-pick
   [deepseek-ai/DeepEP#692](https://github.com/deepseek-ai/DeepEP/pull/692)。
   `st.bulk` 的 size 操作数必须用 64 位寄存器（`"l"` 约束）。
   CUDA <= 13.0 的 ptxas 不支持 32 位形式，否则运行时 JIT 编译
   `dispatch_copy_epilogue` 报错：
   `ptxas error: Arguments mismatch for instruction 'mov'`。

## 4. 编译与安装

```bash
cd /root/DeepEP
TORCH_CUDA_ARCH_LIST="10.0;10.3" EP_NCCL_ROOT_DIR=/usr python setup.py bdist_wheel
pip install --force-reinstall /root/DeepEP/dist/deep_ep-2.1.0+local-cp312-cp312-linux_aarch64.whl
```

## 5. 运行测试

注意：不能在 `/root/DeepEP` 目录里直接跑，Python 会优先导入源码目录下的
`deep_ep/`（不含编译产物 `_C*.so`），覆盖已安装的 wheel，导致
`ModuleNotFoundError: deep_ep._C`。换到其他目录执行：

```bash
cd /tmp
python /root/DeepEP/tests/elastic/test_ep.py --num-processes 4
```

预期结果：144 个用例组合全部通过，退出码 0。
参考性能（4x GPU 单机）：dispatch ~800 GB/s (SU)，combine ~720 GB/s (SU)。
