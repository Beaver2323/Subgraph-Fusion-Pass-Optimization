# T-096 功能与性能测例讲解

> 更新时间：2026-09-11 18:44 CST（UTC+08:00）
> 状态：GPU 3/3、NPU安装态原合同3/3、功能对照7/7通过，1/1单元完成；七元素六臂性能PERF_REGRESSED，额外NPU边界PARTIAL_ALIGNED。4个候选延期。

最新 [根因、数学oracle与候选代码](../issues/REF-e8m0-log2-pattern-native/根因分析.md)；[安装态失败和候选通过分列](../results/current/T-096/npu_blocker_review.json)。

当前学习入口：[代码与GPU/NPU/性能对照](../results/current/T-096/E8M0_讲解.md) / [安装态修复验证报告和命令](../issues/REF-e8m0-log2-pattern-native/修复验证报告.md)。旧失败与候选原件保留，当前状态由installed_resolution接替。

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

功能测例来自社区三条A100 CUDA测试：普通值、`2^e`上方1 ULP边界和gh-178045回归。替换意图是避免software log2在边界舍入到较小整数。上游仍只在CUDA注册且extra-check要求CUDA；这不是NPU产品明确disable。本轮已完成原生阻断、最小设备适配、产品数值域评审和安装态修复，原3例全部通过。

性能派生自社区原七元素正确输入，目标开关为`experimental.config.enable_e8m0_rceil_log2`。OFF/ON均数学正确且前者零命中/后者精确改图，再在空闲NPU上互斥测六臂。host p50/p99慢21.76%/29.25%，Event慢28.32%/27.65%；不是模型端到端，不使用错误的one-ULP OFF算收益。新增配置默认开启是正确性修复，不是依据性能重新打开显式disable优化。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-misc-patterns-e8m0-rceil` | `deferred-hardware-inapplicable-on-a100` | 该分支只在NVIDIA SM100+注册并依赖Blackwell PTX；当前A100 reference会按产品硬件条件跳过。 |
| `AU-misc-patterns-randperm-index` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-misc-patterns-randperm-index-add` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-misc-patterns-randperm-index-full` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
