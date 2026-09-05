# T-079 GPU/reference Runner 操作说明

> 更新时间：2026-09-06 06:43 CST（UTC+08:00）
> 状态：4 个 acceptance units、4 个 direct cases、14 个 variants 及逐单元性能计划已准备，等待 GPU 执行
> 原则：只运行冻结 PyTorch commit 的原生社区方法；失败原样回传，不在 GPU 机器临时修改测试

## 一键执行

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-079 \
  --gpu 2
```

将 `2` 换成所选物理 GPU；默认共享，加 `--wait-gpu` 每 1 秒检查启动条件，独占需加 `--exclusive`。
脚本自动从 `/data/z50063656/tmp` 启动，使用既有 PassGPURef/CUDA 12.6 环境，并为每个 case
建立 fresh process。显存门槛和等卡选项见[通用一键说明](GPU_TASK_RUNNER.md)。

## 静态校验预期

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-079 \
  --validate-only
```

```text
prepared_task_validation=OK task=T-079 units=4 cases=4 variants=14 performance_units=4 guide=valid
reference_plan_validation=OK acceptance_units=4 cases=4 community_tests=4 variants=14 executed_variants=14 non_executed_variants=0
torch_imported=0 gpu_executed=0
gpu_task_validation=OK task=T-079
```

功能测例及其派生性能测例讲解见 `docs/T079_FUNCTION_PERFORMANCE_GUIDE.md`；T-079 没有社区独立
benchmark，不能把 tracker 派生 workload 标成社区性能测例。

## Case 列表

```text
REF-bmm-to-mm-native
REF-cat-slice-cat-native
REF-splitwithsizes-cat-native
REF-cat-splitwithsizes-native
```

单 case 通过 `--case CASE_ID` 选择。完整运行后复制：

```bash
cat /data/z50063656/tmp/t079-reference-results/latest-text-handoff.json
```

只有 4/4 cases 均 `passed` 且 `reference_valid=true` 才能冻结 T-079。
一键入口默认生成可恢复 FX/日志正文的 1.1 handoff；复制、校验和恢复见
[GPU 原文 handoff 指南](GPU_TEXT_HANDOFF.md)。
