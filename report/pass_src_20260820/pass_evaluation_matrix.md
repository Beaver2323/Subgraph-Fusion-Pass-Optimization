# Pass NPU 评估矩阵说明

- 输入清单：`/home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820/pass_inventory.json`
- 记录数：**251**
- 当前阶段：评估合同已生成；P0 gate、different-K、pad family 和 B2 前 18 条已有动态证据。3 条 pad pattern 因 device gate 排除 NPU，产品 verdict 仍为 `unsupported`；T-025测试侧绕过后证明mm/bmm/addmm功能可承接，但T-026三轮p50分别回退`72.65%/65.31%/120.63%`，replacement均为`rejected-performance-regression`。`addmm fusion` 的8个代表配置均为p50正收益，`strict_sum`修复后backward闭环，最终为`supported-beneficial`。different-K的T-023 default-off template在shape-A/unaligned集成p50改善`15.29%/18.04%`，但candidate比baseline均多`270,336 B` peak allocated，T-024又没有找到同时通过显存和task-duration gate的配置；fresh launcher环境合同也尚未闭环，因此保持`conditional-supported-beneficial`。B2中，`fold_reduce`的正确clone方案因性能回退被否决，`cat_to_view_pass`为latency-neutral/resource-beneficial，`fold_cat`为`supported-beneficial`；T-035 将 `fold_where` 关闭为`supported-neutral`。T-036 修复 cat-slice-cat/pad-slice 的零数值误差 alias/stride 缺陷，T-037 又确认二者p50改善`24.00%/31.35%`，均为`supported-beneficial`。矩阵总计240条`not-run`、3条`unsupported`、4条`supported-beneficial`、1条`conditional-supported-beneficial`、1条`supported-neutral`、1条`supported-neutral-resource-beneficial`、1条`supported-pass-disabled-performance-rejected`。

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
