# T-096 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：1 个GPU-ready单元，4 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-096 \
  --gpu 2 \
  --wait-gpu
```

先运行冻结 PyTorch revision 的原生社区测例；只有真实设备/backend/采集阻断才进入最小适配审核。

## AU-misc-patterns-e8m0-rceil-log2

```python
# torch/_inductor/fx_passes/misc_patterns.py:170
log2_val = torch.log2(inp)
ceil_val = torch.ceil(log2_val)
biased = torch.clamp(ceil_val, min=-127, max=127) + 127
# pre-SM100 replacement: 直接读取float32指数和mantissa位域
```

功能测例来自社区三条A100 CUDA测试：普通值、`2^e`上方1 ULP边界和gh-178045回归。替换意图是避免software log2在边界舍入到较小整数。源码当前只在CUDA注册且extra-check要求CUDA；这不是NPU产品明确disable，而是能力待评审。NPU原生阻断和最小适配审核完成前，性能测例仅有设计、不执行。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-misc-patterns-e8m0-rceil` | `deferred-hardware-inapplicable-on-a100` | 该分支只在NVIDIA SM100+注册并依赖Blackwell PTX；当前A100 reference会按产品硬件条件跳过。 |
| `AU-misc-patterns-randperm-index` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-misc-patterns-randperm-index-add` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-misc-patterns-randperm-index-full` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
