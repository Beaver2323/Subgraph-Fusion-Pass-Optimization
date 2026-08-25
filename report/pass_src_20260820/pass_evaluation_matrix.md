# Pass NPU 评估矩阵说明

- 输入清单：`/home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820/pass_inventory.json`
- 记录数：**251**
- 当前阶段：评估合同已生成；P0 与 P1 B2 全部 27 条已有动态或明确 device-gated 证据。T-043～T-046 修复 batch-embedding step/dtype/alias 与 attention schema/meta/重复执行：batch 两个 safe cohort P50 改善 `23.50%/43.90%`、tasks `9→3/13→3`，但 allocated peak 和首次编译增加，记 `supported-neutral-resource-beneficial`；legacy→v3 attention 在 910B2 P50/P99 回退 `4.85%/31.72%`，最终非 A5 停用；fused matmul+relu 在 910B2 为 `not-applicable`，不外推 A5。此前 addmm、different-K、pad family、alias/layout/dtype/mask/Hamming 的结论保持。矩阵总计 231 条 `not-run`、1 条 `not-applicable`、3 条 `unsupported`、7 条 `supported-beneficial`、1 条 `conditional-supported-beneficial`、4 条 `supported-neutral`、2 条 `supported-neutral-resource-beneficial`、2 条 `supported-pass-disabled-performance-rejected`。

## 测试单元

| 类型 | 数量 | 处理方式 |
|---|---:|---|
| `container-coverage` | 25 | registry 本身不单测，由其中 entry 覆盖 |
| `direct-case` | 164 | 为 pattern/custom pass 设计最小触发图 |
| `manual-review` | 21 | 人工确认是否为真实 pass、helper 或 backend hook |
| `observer-stage` | 41 | 验证 pipeline observer、阶段顺序和变更计数 |

## 验证分组

| 分组 | 数量 |
|---|---:|
| `alias-mutation` | 12 |
| `backend-assumption-review` | 5 |
| `backend-sensitive` | 2 |
| `device-specific-review` | 11 |
| `distributed-npu` | 1 |
| `dynamic-shape` | 21 |
| `generic-npu` | 155 |
| `npu-specific` | 44 |

## 执行批次

| 批次 | 数量 |
|---|---:|
| `B10_PIPELINE_INFRA` | 69 |
| `B11_GENERIC` | 27 |
| `B1_NPU_GATES` | 7 |
| `B2_NPU_CUSTOM` | 27 |
| `B3_DVM_MLIR` | 8 |
| `B4_ATTENTION` | 31 |
| `B5_GEMM` | 9 |
| `B6_SPLIT_CAT` | 29 |
| `B7_RANDOM_MISC` | 14 |
| `B8_DEVICE_SPECIFIC` | 25 |
| `B9_DISTRIBUTED` | 5 |

## 填写顺序

1. 人工确认 `applicability_status`，先排除正确 device-gated 的 CPU/CUDA 专用项。
2. 为 `direct-case` 填写 `trigger_case`，并用 observer/counter 证明实际触发。
3. 验证 eager 对齐、generated code、graph break 和 fallback。
4. 只有功能通过后填写统一环境下的性能字段。
5. 对 `unsupported` 或 `supported-regression` 再评估 replacement，不提前指定 Triton。

允许的最终 verdict：`not-run`、`not-applicable`、`environment-blocked`、
`unsupported`、`supported-neutral`、`supported-beneficial`、
`conditional-supported-beneficial`、
`supported-neutral-resource-beneficial`、
`supported-pass-disabled-performance-rejected`、`supported-regression`。
