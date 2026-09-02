# T-076 NPU 正式闭环报告

> 更新时间：2026-09-02 17:42 CST（UTC+08:00）
> 上游 commit：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> GPU reference：`reference-20260901T180826+0800`

## 结论

T-076 首批 5 个冻结 acceptance units 已全部完成 NPU 执行、reference/NPU 对比与正式分类，闭环计数为 `5/5`。

| acceptance unit | 正式结论 | 核心观测 |
| --- | --- | --- |
| `AU-post-grad-mm-plus-mm` | `BEHAVIOR_UNCHANGED` | same-K/different-K 正例与负向 guard 合同成立 |
| `AU-pad-mm-mm` | `EXPECTED_PRODUCT_DIVERGENCE` | NPU 默认 `disable_pad_mm=true`，原图 correctness/stride 正确 |
| `AU-pad-mm-bmm` | `EXPECTED_PRODUCT_DIVERGENCE` | dynamic/static 不执行 padding，autocast backward 回归通过 |
| `AU-pad-mm-addmm` | `EXPECTED_PRODUCT_DIVERGENCE` | dynamic-M 与六种 bias 正确，CUDA-only beta=0 负例为 N/A |
| `AU-post-grad-addmm` | `EXPECTED_PRODUCT_DIVERGENCE` | 实际安装态 `disable_addmm_fusion=True`，正例 `0/0` 是显式 gate；P-018 候选已恢复 `2/4` |

## 执行边界

- 所有 NPU 原生入口均先执行；受 GPU-only suite gate 阻断后才使用 case-specific adapter。
- adapter 只注入 NPU device/backend、必要的 ATEN choice 与 artifact capture，没有修改 PyTorch、torch_npu、Triton 或绕过产品 gate。
- 失败图按图模式规则检查 FX 和 `output_code.py`。post-grad addmm 的首个分歧位于安装态 config gate；原结果错误使用 dirty source overlay 描述运行配置，已纠偏。
- 当前 Pass site-packages 的 addmm gate 默认关闭；`regressions/known_issues.yaml` 中原回归已移出 open issues 并保留 reclassified 审计记录。
- P-018 独立 wheel 在同一冻结 PyTorch commit 上完成精确上游复验：matrix/vector 正例均为 `2/4`，non-expandable/batched/Python/symbolic scalar 负例均为 `0/0`；正例生成两个 extern addmm。
- 每个 variant 的意图、源码位置、GPU/NPU 行为已写入 comparison JSON；带关键源码块的学习导读见 `report/t076_pattern_gpu_npu_guide_20260902.md`。
- 正式结果位于 `results/current/`；原始运行产物保留在 `/home/z50063656/tmp/t076-npu-results/`，不提交重型 artifact。

## 后续

T-076 任务已完成：1 个 `BEHAVIOR_UNCHANGED`、4 个 `EXPECTED_PRODUCT_DIVERGENCE`，当前没有 open `NPU_REGRESSION`。P-018 是已验证的默认启用候选，不需要新增 lowering；是否将候选并入正式产品由后续变更评审决定。T-077 已准备 GPU direct reference suite，等待 GPU 人工执行，其完整流程包含条件 repair 阶段。
