# 历史文档归档说明

> 归档整理时间：2026-08-31 17:50 CST（UTC+08:00）
> 状态：只读历史证据；当前工作以仓库根目录 README/TODO/WORKFLOW 和 `docs/CURRENT_STATUS.md`
> 为准。

本目录保存 2026-08-31 工作线校准前的检查点、计划、交接和总览。文件被移动到这里是为了减少
根目录重复入口，不表示其结果无效，也不授权改写其原始环境或 verdict。

## 目录

### `background/`

- `README_20260829.md`：8 月 29 日前累计写入根 README 的完整任务叙述；
- `AUDIT_OVERVIEW_20260829.md`：旧审计总览；
- `CURRENT_STATUS_20260829.md`：旧环境、阶段和结果总览。

三者内容高度重叠，已由新的 `README.md`、`docs/CURRENT_STATUS.md` 和 `docs/HISTORY.md` 分工取代。

### `checkpoints/`

- `PAUSED_CHECKPOINT_20260821.md`；
- `PAUSED_CHECKPOINT_20260826_P013.md`。

这些文件用于复现旧暂停点，不是当前恢复入口。

### `handoffs/`

- `HANDOFF_20260828_PASS_ENV.md`：Conda Pass 环境和 T-074 第一版交接；

环境合同仍有参考价值，但当前优先级以 2026-08-31 tracker 文档为准。

### `plans/`

- `p0_case_design.md`、`p1_batch_design.md`：default/custom/attention 历史批次设计；
- `replacement_plan.md`：旧替代实现计划；
- `triton_experimental_migration_20260826.md`：feature-family 迁移基线。

这些计划保留为 previous-phase evidence；不得把其中的“下一步”当成当前 TODO。

## 语言与时间戳说明

活动文档统一为中文并带 2026-08-31 时间戳。归档文件为了保留提交时的真实内容，可能包含旧的
英文标题、旧时间口径或旧路径；其归档时间由本索引统一记录，不对正文做机械重写。
