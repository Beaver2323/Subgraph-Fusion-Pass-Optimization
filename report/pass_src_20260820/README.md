# 2026-08-20 Pass Inventory 与评估矩阵（历史证据）

> 边界更新时间：2026-09-06 07:21 CST（UTC+08:00）
> 状态：`historical-non-counting`。本目录保留早期 registration/inventory、default backend、
> torch_npu custom、DVM、MLIR 与初期 experimental 调查，不能直接产生当前 tracker verdict。

## 为什么仍包含非 `triton_experimental` 项

这份清单是在项目转为 community-native compatibility tracker 之前建立的源码全景。251 行记录中
既有 PyTorch upstream 的 pre-grad/joint-graph/post-grad registration，也有 Inductor extension、
torch_npu custom pass、DVM、MLIR 和早期 `triton_experimental` 项。

- upstream pre/joint/post-grad registration 不是某个 NPU backend；它们是待验证的社区优化合同；
- CUDA/GPU 的 `inductor-default` 是 reference 端，保留是有意设计；
- default/DVM/MLIR/custom 的旧 NPU 动态结果仅为历史证据，不得迁移成
  `triton_experimental` verdict。

## 当前应使用什么

- 人工审核后的活动单元与当前状态：
  [`../current_acceptance_unit_matrix.md`](../current_acceptance_unit_matrix.md)
- 机器可读当前矩阵：
  [`../current_acceptance_unit_matrix.csv`](../current_acceptance_unit_matrix.csv)
- 当前执行规则：[`../../WORKFLOW.md`](../../WORKFLOW.md)
- 活动 manifest：[`../../upstream/`](../../upstream/)
- 正式 NPU/comparison 结果：[`../../results/current/`](../../results/current/)

`pass_evaluation_matrix.csv` 保持原始 251 行不变，以免改写历史证据；其 Markdown 说明已增加同样的
历史边界提示。旧矩阵可用于学习字段设计、追溯早期实验和发现候选，不能作为当前完成分母。
