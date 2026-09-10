# 当前 Acceptance Unit 兼容性矩阵

> 生成时间：2026-09-10T07:33:34+08:00
> 数据源：`upstream/*manifest.yaml`、`results/current/` 与逐任务性能计划/汇总。
> 后端边界：GPU reference 固定为 `inductor-default`；NPU 动态验证、比较、修复验证与性能固定为 `triton_experimental`。
> 历史 251 行 registration 矩阵不参与本表 verdict；其用途与边界见 `report/archive/legacy-20260820-0828/pass_src_20260820/README.md`。

## 状态摘要

- 活动 acceptance units：**33**；已冻结 reference：**33**；存在覆盖扩展未闭环：**1**。
- 已形成 NPU/comparison：**33**；已有正式性能处置：**33**；其余为性能计划态。
- `comparison`/性能处置数量只说明已登记 variants；存在 pending extension 的单元必须以“覆盖”和“当前阶段”列为准，不能外推为全域闭环。
- 当前 NPU 结果实际观测 backend：`triton_experimental`。
- 本表汇总已登记结论，不代表严格历史再认证通过；T-076/T-077 的独立补证状态见 [最新审计](../results/audits/latest.json)。
- `npu_execution_status=failed` 不自动表示数值错误；例如产品 gate 关闭时，目标命中失败可与原图 correctness 通过同时成立，应结合 comparison verdict 阅读。
- 社区对齐列必须区分完全对齐、部分对齐、预期后端差异、需修复和待复核；旧结果缺少显式字段时一律显示待复核，不从既有 PASS 自动外推。

## 单元矩阵

| T | Acceptance unit | Stage | 覆盖 | Reference | NPU backend | NPU 执行 | Correctness | Comparison | Repair | 社区对齐/处置 | 性能处置 | 当前阶段 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-076 | AU-post-grad-mm-plus-mm | post_grad | fully-covered | frozen-reference-valid | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured / representative-beneficial-with-neutral-layouts | functional-comparison-closed |
| T-076 | AU-pad-mm-mm | joint_graph | fully-covered | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-pad-mm-bmm | joint_graph | fully-covered | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-pad-mm-addmm | joint_graph | fully-covered | frozen-reference-valid | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | not-required-explicitly-disabled / product-disabled-performance-exempt | functional-comparison-closed |
| T-076 | AU-post-grad-addmm | post_grad | fully-covered | frozen-reference-valid | triton_experimental | failed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured / beneficial-with-host-tail-monitor | functional-comparison-closed |
| T-077 | AU-apply-gumbel-max-trick | pre_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_IMPROVED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | passed / PERF_IMPROVED | functional-comparison-closed |
| T-077 | AU-b2b-gemm | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | capability-assessed-no-effective-template / CAPABILITY_REJECTED_NO_EFFECTIVE_TEMPLATE | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-bmm | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-mm | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | NPU_REGRESSION | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-077 | AU-decompose-mem-bound-mm-decompose-addmm | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-candidate-rejected / PERF_REGRESSED | functional-comparison-closed |
| T-078 | AU-post-grad-fuse-addcdiv-to-fma | post_grad | verified=5; pending=fp16-value1-bitwise-regression[reference=pending-derived-dtype-and-value-neighbor;npu=fixed-on-device-20260909-bitwise-counter0] | valid-reference-suite-with-pending-extension | triton_experimental | passed | passed-for-existing-variants | PERF_NEUTRAL-for-existing-variants | verified-for-existing-variants | PARTIAL_ALIGNED / 保留经过位级验证的 NPU 专属 FP16 value!=1 addcdiv lowering；value=1 通过 NPU FP16 aten.div.Tensor 舍入边界修复，保持 counter=0 且不伪造 FMA pattern 命中。 | measured-retained-verified-variants-only / PERF_NEUTRAL-for-existing-variants | coverage-extension-gpu-reference-pending |
| T-078 | AU-post-grad-reuse-partial | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_MIXED | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-retained-shape-dependent / PERF_MIXED | functional-comparison-closed |
| T-078 | AU-post-grad-unfuse-bias-add-to-pointwise | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-candidate-rejected-explicitly-disabled / PERF_REGRESSED | functional-comparison-closed |
| T-078 | AU-post-grad-unfuse-bias-baddbmm-to-pointwise | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_MIXED | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-selective-product-gate / PERF_MIXED | functional-comparison-closed |
| T-079 | AU-joint-graph-bmm-to-mm | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-regressed-product-disabled / PERF_REGRESSED | functional-comparison-closed |
| T-079 | AU-post-grad-cat-slice-cat | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_IMPROVED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-improved-retained / PERF_IMPROVED | functional-comparison-closed |
| T-079 | AU-post-grad-splitwithsizes-cat-replace | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_IMPROVED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-improved-retained / PERF_IMPROVED | functional-comparison-closed |
| T-079 | AU-post-grad-cat-splitwithsizes-replace | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_IMPROVED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-improved-retained / PERF_IMPROVED | functional-comparison-closed |
| T-080 | AU-joint-graph-scatter-upon-const-tensor | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | verified | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-regressed-product-disabled / PERF_REGRESSED | functional-comparison-closed |
| T-080 | AU-post-grad-prepare-softmax | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | EXPECTED_PRODUCT_DIVERGENCE | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | exempt-explicit-product-lowering-disable / PERF_EXEMPT | functional-comparison-closed |
| T-080 | AU-post-grad-move-constructors-to-gpu | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | PERF_NEUTRAL | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-neutral-retain-enabled / PERF_NEUTRAL | functional-comparison-closed |
| T-081 | AU-joint-graph-constant-fold-uniform-value | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-improved / PERF_IMPROVED | functional-comparison-closed |
| T-081 | AU-joint-graph-pointless-convert | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-neutral / PERF_NEUTRAL | functional-comparison-closed |
| T-082 | AU-joint-graph-pointless-permute-pair | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-neutral / PERF_NEUTRAL | functional-comparison-closed |
| T-082 | AU-joint-graph-pointless-view-pair | joint_graph | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-neutral / PERF_NEUTRAL | functional-comparison-closed |
| T-083 | AU-post-grad-bucket-all-gathers | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-regressed / PERF_REGRESSED | functional-comparison-closed |
| T-083 | AU-post-grad-bucket-all-reduce | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-improved / PERF_IMPROVED | functional-comparison-closed |
| T-083 | AU-post-grad-bucket-reduce-scatters | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PENDING_REVIEW / 保留原功能/性能结论，但不得据此外推为完全社区对齐 | measured-mixed / PERF_MIXED | functional-comparison-closed |
| T-084 | AU-post-grad-dedup-reduce-scatters | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | ALIGNED_WITH_ADAPTER / 设备/进程组作最小适配；合同与社区一致，性能只按NPU两rank实测判定。 | measured-improved / PERF_IMPROVED | formally-closed |
| T-085 | AU-post-grad-overlap-scheduling-device-put-sync | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | ALIGNED_WITH_ADAPTER / 保留社区安全改写；带宽来源是后端最小适配，不把高方差数据写成收益。 | measured-mixed-high-variance / PERF_MIXED | formally-closed |
| T-085 | AU-post-grad-partitioned-scatter-optimization | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PARTIALLY_ALIGNED / 能力路径正确但不改变默认关闭；NPU显存探针替换CUDA专用探针。 | measured-regressed / PERF_REGRESSED | formally-closed |
| T-085 | AU-post-grad-pointless-cumsum | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | PARTIALLY_ALIGNED / 保留上游CPU/CUDA行为，仅在triton_experimental NPU通过可逆gate关闭。 | measured-regressed / PERF_REGRESSED | formally-closed |
| T-086 | AU-post-grad-reinplace-inplaceable-ops | post_grad | fully-covered | valid-reference-suite | triton_experimental | passed | passed | BEHAVIOR_UNCHANGED | not-needed | ALIGNED_WITH_BACKEND_LOWERING_DIFFERENCE / 保留功能改写；性能结论为混合，不外推稳定收益。 | measured-mixed / PERF_MIXED | formally-closed |

## 使用说明

- 本 Markdown 便于阅读；完整字段、证据路径和生成时间以同目录 CSV 为准。
- `reference_backend=inductor-default` 表示 CUDA/GPU 对照端，不能据此声称 NPU 使用了 default backend。
- 只有 `observed_npu_backend=triton_experimental` 的动态结果可进入当前 NPU comparison。空值表示尚未运行，不表示可使用其他 backend。
- 性能证据路径指向 `results/current/` 时表示已有处置；指向 `upstream/*_performance_plan.yaml` 时只表示测量合同已准备。
- 社区对齐的完整已对齐/差异/未决范围保存在 CSV；`source=legacy-missing-explicit` 表示旧结果仍需显式再认证。
- 修改 manifest/result 后运行 `python scripts/generate_current_acceptance_matrix.py --write` 更新，再运行 `--check` 做一致性校验。
