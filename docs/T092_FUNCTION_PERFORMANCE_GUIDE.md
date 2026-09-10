# T-092 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，5 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## 功能测例

本批经审核没有可冻结的原生GPU直接合同。可做零设备静态校验，但一键入口会拒绝实际空跑：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-092 \
  --validate-only
```

这不是SKIP或PASS；所有旧ID均保留为非计数延期记录。

## 性能测例

没有通过GPU功能门禁的合同，因此不派生性能测例、不制造ON路径，性能状态为不适用。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-split-cat-normalize-unbind-default` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-split-cat-remove-split-with-size-one` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-split-cat-replace-einsum-to-pointwise` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-split-cat-simplify-split-cat` | `deferred-mismapped-or-cpu-only-tests` | 列出的Aten测例实际归属post-grad merge_split_cat_aten；其余pre-grad测例为CPU，不能证明本handler的GPU入口。 |
| `AU-split-cat-split-cat-to-slices` | `deferred-cpu-multi-pattern-test` | test_split_cat_new_patterns使用CPU并同时覆盖多个handler，缺少独立GPU归属。 |
