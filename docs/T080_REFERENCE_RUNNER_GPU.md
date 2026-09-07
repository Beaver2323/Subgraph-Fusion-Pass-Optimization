# T-080 GPU/reference Runner 操作说明

> 更新时间：2026-09-07 08:32 CST（UTC+08:00）
> 状态：GPU 13/13 direct cases、13/13 variants 已回传、逐项复核并冻结；下一步为 NPU
> `triton_experimental` 功能/命中验证
> 原则：reference 阶段只跑社区默认功能规模；`DO_PERF_TEST=1` 的社区性能路径留到功能/NPU 门禁后

## 一键执行

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-080 \
  --gpu 2
```

默认共享，允许已有计算进程；加 `--wait-gpu` 每 1 秒检查启动条件，独占需加 `--exclusive`。
显存门槛、等卡超时和固定结果入口见[通用一键说明](GPU_TASK_RUNNER.md)。

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

完整运行后只需读取控制台打印的 `handoff_upload_mode` 和 `handoff_upload_input`。未超限时复制：

```bash
cat /data/z50063656/tmp/t080-reference-results/latest-text-handoff.json
```

超过 96 KiB 时脚本自动生成分片，上传 `handoff_upload_input` 所在目录的 `manifest.json` 与全部
`part-*.json`；无需先尝试大文件，也无需手工重新导出。

只有 13/13 cases 均 `passed` 且 `reference_valid=true` 才能冻结 T-080。即使社区方法包含性能代码，
本轮也不设置 `DO_PERF_TEST=1`；后续仅在 NPU `triton_experimental` 功能命中、correctness 与
artifact 门禁通过后，才运行同来源 OFF/ON 性能对照。

一键入口默认生成可恢复 FX 与关键 case 正文的 1.3 review handoff；复制、校验和恢复见
[GPU 原文 handoff 指南](GPU_TEXT_HANDOFF.md)。

各 case 的输入、guard、社区 benchmark 复用边界和 compiled OFF/ON 设计见
`docs/T080_FUNCTION_PERFORMANCE_GUIDE.md` 与 `upstream/t080_performance_plan.yaml`。
