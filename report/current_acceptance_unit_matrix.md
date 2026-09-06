# 当前 Acceptance Unit 兼容性矩阵

> 生成时间：2026-09-06T07:21:22+08:00
> 数据源：`upstream/*manifest.yaml`、`results/current/` 与逐任务性能计划/汇总。
> 后端边界：GPU reference 固定为 `inductor-default`；NPU 动态验证、比较、修复验证与性能固定为 `triton_experimental`。
> 历史 251 行 registration 矩阵不参与本表 verdict；其用途与边界见 `report/pass_src_20260820/README.md`。

## 状态摘要

- 活动 acceptance units：**21**；已冻结 reference：**10**；等待 GPU reference：**11**。
- 已形成 NPU/comparison：**10**；已有正式性能处置：**10**；其余为性能计划态。
- 当前 NPU 结果实际观测 backend：`triton_experimental`。
- `npu_execution_status=failed` 不自动表示数值错误；例如产品 gate 关闭时，目标命中失败可与原图 correctness 通过同时成立，应结合 comparison verdict 阅读。

## 单元矩阵

| T | Acceptance unit | Stage | Reference | NPU backend | NPU 执行 | Correctness | Comparison | Repair | 性能处置 | 当前阶段 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-076 | AU-post-grad-mm-plus-mm | post_grad | frozen-reference-valid | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | measured / representative-beneficial-with-neutral-layouts | functional-comparison-closed |
| T-076 | AU-pad-mm-mm | joint_graph | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-pad-mm-bmm | joint_graph | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-pad-mm-addmm | joint_graph | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-post-grad-addmm | post_grad | frozen-reference-valid | triton_experimental | failed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | measured / beneficial-with-host-tail-monitor | functional-comparison-closed |
| T-077 | AU-apply-gumbel-max-trick | pre_grad | valid-reference-suite | triton_experimental | passed | passed | PERF_IMPROVED | not-needed | passed / PERF_IMPROVED | functional-comparison-closed |
| T-077 | AU-b2b-gemm | post_grad | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | capability-assessed-no-effective-template / CAPABILITY_REJECTED_NO_EFFECTIVE_TEMPLATE | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-bmm | post_grad | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-mm | post_grad | valid-reference-suite | triton_experimental | passed | passed | NPU_REGRESSION | verified | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-addmm | post_grad | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-078 | AU-post-grad-fuse-addcdiv-to-fma | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-capability-pending / planned | awaiting-gpu-reference |
| T-078 | AU-post-grad-reuse-partial | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated / planned | awaiting-gpu-reference |
| T-078 | AU-post-grad-unfuse-bias-add-to-pointwise | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-capability-pending / planned | awaiting-gpu-reference |
| T-078 | AU-post-grad-unfuse-bias-baddbmm-to-pointwise | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-capability-pending / planned | awaiting-gpu-reference |
| T-079 | AU-joint-graph-bmm-to-mm | joint_graph | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-capability-pending / planned | awaiting-gpu-reference |
| T-079 | AU-post-grad-cat-slice-cat | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated / planned | awaiting-gpu-reference |
| T-079 | AU-post-grad-splitwithsizes-cat-replace | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated / planned | awaiting-gpu-reference |
| T-079 | AU-post-grad-cat-splitwithsizes-replace | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated / planned | awaiting-gpu-reference |
| T-080 | AU-joint-graph-scatter-upon-const-tensor | joint_graph | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-community-benchmark-ready / planned | awaiting-gpu-reference |
| T-080 | AU-post-grad-prepare-softmax | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-community-benchmark-ready-capability-pending / planned | awaiting-gpu-reference |
| T-080 | AU-post-grad-move-constructors-to-gpu | post_grad | pending-reference | 待测（要求 triton_experimental） | not-run | not-run | not-run | not-run | planned-gated-capability-pending / planned | awaiting-gpu-reference |

## 使用说明

- 本 Markdown 便于阅读；完整字段、证据路径和生成时间以同目录 CSV 为准。
- `reference_backend=inductor-default` 表示 CUDA/GPU 对照端，不能据此声称 NPU 使用了 default backend。
- 只有 `observed_npu_backend=triton_experimental` 的动态结果可进入当前 NPU comparison。空值表示尚未运行，不表示可使用其他 backend。
- 性能证据路径指向 `results/current/` 时表示已有处置；指向 `upstream/*_performance_plan.yaml` 时只表示测量合同已准备。
- 修改 manifest/result 后运行 `python scripts/generate_current_acceptance_matrix.py --write` 更新，再运行 `--check` 做一致性校验。
