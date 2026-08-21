# Pass NPU 评估矩阵说明

- 输入清单：`/home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820/pass_inventory.json`
- 记录数：**251**
- 当前阶段：评估合同已生成；7 条 P0 gate 记录已有动态证据。3 条 pad pattern 因 device gate 排除 NPU，产品 verdict 仍为 `unsupported`；T-025测试侧绕过后证明mm/bmm/addmm功能可承接，但T-026三轮p50分别回退`72.65%/65.31%/120.63%`，replacement均为`rejected-performance-regression`。`addmm fusion` 的8个代表配置均为p50正收益，`strict_sum`修复后backward闭环，最终为`supported-beneficial`。`mm_plus_mm` same-K experimental网格8/8功能正确、6个beneficial；different-K的T-014至T-022已关闭三dtype、真实转置、dynamic/backward、扩展性能、正式设计和large hold。T-023已安装NPU-only、default-off、extern fallback-first template wheel，row/column、AOTAutograd和五类negative通过；shape-A/unaligned集成p50改善`15.29%/18.04%`，但candidate比baseline均多`270,336 B` peak allocated，根因为`65,536 B × 6` Triton Ascend workspace。T-024没找到同时通过显存和task-duration gate的配置。fresh launcher仍需匹配PyTorch C++20、Triton Ascend、torch_npu与CANN headers，所以verdict为`conditional-supported-beneficial`，默认关闭。矩阵总计246条`not-run`、3条`unsupported`、1条`supported-beneficial`、1条`conditional-supported-beneficial`。

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
`supported-regression`。
