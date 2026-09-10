# T-094 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，5 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## 功能测例

本批经审核没有可冻结的原生GPU直接合同。可做零设备静态校验，但一键入口会拒绝实际空跑：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-094 \
  --validate-only
```

这不是SKIP或PASS；所有旧ID均保留为非计数延期记录。

## 性能测例

没有通过GPU功能门禁的合同，因此不派生性能测例、不制造ON路径，性能状态为不适用。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-pre-grad-efficient-conv-bn-eval` | `deferred-structural-dispatcher-merged-into-t098` | 这是pre_grad对efficient_conv_bn_eval_pass的调度入口；动态合同在T-098按实际handler计一次，避免重复分母。 |
| `AU-pre-grad-fuse-conv-bn` | `deferred-mismapped-community-test` | 旧索引指向test_basic内部函数，实际测的是efficient_conv_bn_eval handlers，不是该legacy helper。 |
| `AU-pre-grad-linear-permute-fusion` | `deferred-pure-cpu-fx-helper-test` | test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。 |
| `AU-pre-grad-permute-linear-fusion` | `deferred-pure-cpu-fx-helper-test` | test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。 |
| `AU-pre-grad-permute-matmul-fusion` | `deferred-pure-cpu-fx-helper-test` | test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。 |
