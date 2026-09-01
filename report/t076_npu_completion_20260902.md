# T-076 NPU 正式闭环报告

> 生成时间：2026-09-02 05:35 CST（UTC+08:00）
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
| `AU-post-grad-addmm` | `NPU_REGRESSION` | matrix/vector 正例数值正确，但 target counter 由 GPU `2/4` 变为 NPU `0/0` |

## 执行边界

- 所有 NPU 原生入口均先执行；受 GPU-only suite gate 阻断后才使用 case-specific adapter。
- adapter 只注入 NPU device/backend、必要的 ATEN choice 与 artifact capture，没有修改 PyTorch、torch_npu、Triton 或绕过产品 gate。
- 失败图按图模式规则检查 FX 和 `output_code.py`。post-grad addmm 的首个分歧位于 pattern/replacement，已写入 `regressions/known_issues.yaml`。
- 正式结果位于 `results/current/`；原始运行产物保留在 `/home/z50063656/tmp/t076-npu-results/`，不提交重型 artifact。

## 后续

T-076 任务已完成；`AU-post-grad-addmm` 的产品修复作为独立 repair 工作线继续。T-077 已准备 GPU direct reference suite，等待 GPU 人工执行。
