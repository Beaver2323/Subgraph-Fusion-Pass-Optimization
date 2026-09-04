# T-080 GPU/reference Runner 操作说明

> 更新时间：2026-09-04 09:08 CST（UTC+08:00）
> 状态：3 个 acceptance units、13 个 direct cases、13 个 variants 及逐单元性能计划已准备，等待 GPU 执行
> 原则：reference 阶段只跑社区默认功能规模；`DO_PERF_TEST=1` 的社区性能路径留到功能/NPU 门禁后

## 一键执行

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-080 \
  --gpu 2
```

## 静态校验预期

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-080 \
  --validate-only
```

```text
prepared_task_validation=OK task=T-080 units=3 cases=13 variants=13 performance_units=3 guide=valid
reference_plan_validation=OK acceptance_units=3 cases=13 community_tests=13 variants=13 executed_variants=13 non_executed_variants=0
torch_imported=0 gpu_executed=0
gpu_task_validation=OK task=T-080
```

## Case 列表

```text
REF-scatter-const-3d-native
REF-scatter-const-non-last-dim-native
REF-scatter-const-negative-dim-native
REF-scatter-const-short-index-negative-native
REF-scatter-const-dense-negative-native
REF-scatter-const-nonconst-negative-native
REF-scatter-const-dtype-regression-native
REF-scatter-const-cross-entropy-e2e-native
REF-prepare-softmax-fast-math-native
REF-prepare-softmax-signed-zero-native
REF-prepare-softmax-community-perf-native
REF-move-constructors-arange-native
REF-move-constructors-index-put-negative-native
```

完整运行后复制：

```bash
cat /data/z50063656/tmp/t080-reference-results/latest-text-handoff.json
```

只有 13/13 cases 均 `passed` 且 `reference_valid=true` 才能冻结 T-080。即使社区方法包含性能代码，
本轮也不设置 `DO_PERF_TEST=1`；后续仅在 NPU `triton_experimental` 功能命中、correctness 与
artifact 门禁通过后，才运行同来源 OFF/ON 性能对照。

各 case 的输入、guard、社区 benchmark 复用边界和 compiled OFF/ON 设计见
`docs/T080_FUNCTION_PERFORMANCE_GUIDE.md` 与 `upstream/t080_performance_plan.yaml`。
