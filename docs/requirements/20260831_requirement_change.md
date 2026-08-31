# 2026-08-31 Triton Experimental Pass Tracker 需求变更

> 需求接收时间：2026-08-31 17:50 CST（UTC+08:00）
> 状态：已完成文档口径校准和仓库结构收束，后续从 T-075 继续。
> 原始输入：`/home/z50063656/Pass/831需求变更.md`、
> `831TODO_triton_experimental_pass_tracker.md`、
> `831WORKFLOW_triton_experimental_pass_tracker.md`。

## 1. 需求目标

暂停新增 pass 测例和大规模 GPU/NPU 执行，先校准 README、TODO、WORKFLOW、范围/代码地图、
T-074 及后续任务规划，使仓库从一次性 pass 审计升级为持续兼容性 tracker。

活动主线固定为：

```text
PyTorch upstream community tests
→ upstream/GPU expected behavior
→ acceptance unit
→ NPU triton_experimental validation
→ comparison / failure classification
→ repair
→ regression tracking
```

## 2. 概念合同

必须区分：

- registration site/candidate：源码静态 inventory 和 coverage 查漏入口；
- pattern：PatternMatcher 实际注册/匹配的模式；
- acceptance unit：一个需独立验证的 upstream optimization contract；
- variant：同一 contract 下需分开记录的重要能力分支；
- community test：expected behavior 的主要事实源。

禁止假设：

```text
registration == pattern == pass == acceptance unit
```

静态 registration candidate 数量不得直接写成“需要验证的 Pass 总数”。

## 3. Acceptance unit 证据要求

每个冻结单元尽可能记录：

- upstream source/commit；
- registration/pattern evidence；
- upstream community tests；
- GPU/reference baseline；
- NPU trigger 和 correctness；
- lowering/scheduler/codegen；
- fallback/graph-break/runtime path；
- 通过功能门禁后的 performance；
- final verdict、first divergence、failure layer 和 repair status。

## 4. 双机工作流

GPU 机器没有 Agent，只运行仓库预生成的批量脚本、保存结构化 artifacts，并由人工完成必要的
Git/打包交接。NPU 机器作为控制节点，负责 mapping、runner 生成、NPU 执行、artifacts 分析、
failure classification、修复、回归和报告。

不得设计为同一个 Agent 必须同时控制 GPU/NPU 两台机器。

## 5. TODO 优先顺序

1. 收敛 community-test/acceptance-unit mapping；
2. 建立 manifest 与 registration-test-unit 多对多 map；
3. 审核 no-test-found 和 indirect；
4. 建立统一 result schema；
5. 生成 GPU/reference runner；
6. 获取 GPU artifacts；
7. 执行 NPU runner；
8. compare/verdict；
9. failure taxonomy 与 repair queue；
10. upstream 增量检测和 regression tracking。

## 6. 历史证据保护

default Inductor backend、torch_npu custom pass、pass inventory、旧 benchmark 和已完成实验结果
不得删除或改写其真实环境。它们统一标为 historical/previous-phase/archived evidence。

2026-08-29 之后的 active mainline 是 community-native Inductor compatibility on NPU
`triton_experimental`。

## 7. T-074 决策

- candidate inventory 保留，不立即重扫；
- direct/indirect/no-test-found 分类继续作为 review 信号；
- 188/158 只保留 provisional 口径，不冻结分母；
- acceptance-unit grouping 需要按 community contract 人工审阅；
- 无法确认的映射标记 `needs-review`，不得猜测；
- 审核后如需合并/拆分，再生成可追溯的 v2，不覆盖 v1。

## 8. 本次交付决定

- 活动文档全部使用中文并标注时间戳；
- 根目录只保留 README/TODO/WORKFLOW；
- 当前文档归入 `docs/`，历史文档归入 `docs/archive/`；
- `report/` 保留原始实验事实并增加导航；
- 下一任务为 T-075 acceptance-unit schema 与首批 5 单元人工复核；
- T-075 完成后才进入 T-076 GPU/reference runner。
