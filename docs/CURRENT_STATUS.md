# 当前状态与 2026-08-31 工作线校准结论

> 更新时间：2026-08-31 17:50 CST（UTC+08:00）
> 校准输入：`831需求变更.md`、`831TODO_triton_experimental_pass_tracker.md`、
> `831WORKFLOW_triton_experimental_pass_tracker.md`。
> 当前阶段：文档校准与仓库结构收束；尚未启动新的大规模 GPU/NPU 实验。

## 1. 总结

8 月 31 日需求与 8 月 29 日主线方向一致，都要求从 T-056/T-074 的静态 inventory 转向
PyTorch community-native Inductor compatibility。需要修正的不是基础候选数据，而是任务主键、
事实源、TODO 顺序和仓库信息架构：

- 主键从 registration candidate 改为经过人工审核的 acceptance unit；
- 事实源从源码扫描优先改为 community test + GPU/reference baseline 优先；
- T-074 从“待验证 pass 清单”重新定位为 inventory 与 provisional mapping；
- 动态顺序从“逐个 NPU case”改为 mapping → GPU baseline → NPU → compare → repair；
- GPU 无 Agent，NPU 为控制节点；
- 历史 default/custom/feature-family 结果保留，但不计入新主线完成率。

## 2. 工作线吻合性

| 主题 | 8 月 29 日已有工作 | 8 月 31 日要求 | 校准决定 |
| --- | --- | --- | --- |
| 主线 | community-native pass/pattern → tests → NPU | community tests → acceptance unit → GPU/NPU | 方向吻合，补齐 GPU/reference 与 tracker 层 |
| 静态 inventory | 203 主候选 + 4 控制行 | registration 只能辅助查漏 | 保留 207 行，不称为 Pass 数量 |
| 验收单元 | 188 个、158 provisional | contract 才是基本单位 | 保留当前版本，但必须人工复核后冻结 |
| 测试事实源 | 已映射 community tests，仍夹杂源码扫描中心表述 | community tests 为主要事实源 | README/TODO/WORKFLOW 全部改为 test-first |
| 动态测试 | 首批 5 个拟做 NPU 最小迁移 | 先 GPU baseline，再 NPU comparison | 插入 manifest/schema 和 GPU runner 阶段 |
| 双机 | 旧流程主要围绕 NPU | GPU 无 Agent、NPU 有 Agent | 明确固定脚本与 artifacts 交接 |
| 历史结果 | 保留 old Benchmark/isolated venv | historical/archived evidence | 归档，不删除、不升级 verdict |

## 3. T-074 复核结论

### 3.1 可以继续复用的部分

- 203 条 `inherited-upstream-needs-dynamic-validation` 与 4 条显式关闭控制项的集合边界清楚；
- 207 行拥有唯一 `candidate_id`，旧 `record_id` 重复也已显式保留；
- candidate CSV 已区分 implementation、registration、config、community tests、coverage、
  historical evidence 和动态状态；
- `direct-trigger-test`、`indirect-regression-test`、`no-test-found` 的定义适合作为 mapping
  review 信号；
- registry container 和 extension hook 已被识别为不直接进入 provisional denominator；
- 所有动态状态保持 `not-run-current-Pass`，没有把静态映射伪装成 NPU verdict。

### 3.2 仍需审核的部分

当前 acceptance-unit ID 主要按 `source stem + normalized name` 生成，只对少数已知 semantic group
做显式合并。这个规则能提供第一版去重，但不能自动证明一个组就是一个 upstream optimization
contract：

- 同名 registration 可能生成多个具有独立 expected behavior 的 pattern variants；
- 多个 source/registration 可能共同实现一个 contract；
- 一个 community regression test 可能只做间接覆盖，不能证明具体 transformation；
- 仅靠测试名/token overlap 不能冻结 direct contract；
- 自动聚合中有 15 个多 candidate 单元，9 个还同时包含多个 pass/pattern 名称，需要优先审阅。

因此 188 个单元和 158 个 `yes-provisional` 只能描述 T-074 v1 的 heuristic 输出，不能作为
项目规模、成功率或剩余 Pass 总数。

### 3.3 是否重新生成数据

本轮不重新运行或覆盖 T-074：

1. registration candidate inventory 没有因 8 月 31 日术语校准失效；
2. 直接重跑同一 heuristic 只会得到相同 188/158，不能解决 contract 定义问题；
3. 应先冻结 acceptance-unit schema，并人工复核 community test/contract/variant；
4. 审核发现合并、拆分或 schema 字段缺口后，再生成 T-074 v2/manifest，并保留 v1 作为输入证据。

当前数量因此**没有变化**：207 candidate rows、188 provisional units、158 provisional eligible、
5 个首批人工映射、正式闭环 0。未来人工审核可以改变 acceptance-unit 数量，这是预期的可审计
修正，不是数据回归。

## 4. 已修正的概念问题

| 旧表述风险 | 问题 | 新口径 |
| --- | --- | --- |
| “每个 pass 都有……” | 暗示 inventory 行与独立任务一一对应 | 每个冻结 acceptance unit 承载完整证据 |
| “203 个上游候选/Pass” | 容易把 registration candidate 当 Pass 总数 | 203 是主候选行，只用于 inventory |
| “188 个去重 Pass” | heuristic grouping 尚未证明 contract | 188 个 provisional acceptance units |
| “先逐个写 NPU case” | 绕过 community contract 和 reference baseline | 先 mapping/manifest/GPU，再 NPU |
| “community test 是映射字段之一” | 弱化主要事实源地位 | community test + GPU baseline 定义 expected behavior |
| “compile 成功即支持” | 可能实际走 extern/fallback | 必须记录 runtime path 与完整链路 |
| “正确即可测性能并宣称收益” | 未隔离 pass 与 fallback | correctness 门禁后做 fresh-process OFF/ON paired |

历史报告中的原始统计和当时表述不机械改写；它们通过 archive/index 明确标为 previous phase。

## 5. 仓库结构调整

根目录从多份互相重复的总览/计划收束为：

```text
README.md       当前定位与导航
TODO.md         唯一活动任务清单
WORKFLOW.md     唯一活动执行流程
docs/           当前状态、范围、指南、变更控制、历史与归档
report/         不可改写的实验事实和 T-074 数据
```

旧 README、旧 current status、旧 audit overview、检查点、计划和交接全部移动到
`docs/archive/`，没有删除。`docs/HISTORY.md` 保留成果索引，`report/README.md` 提供证据导航。

## 6. 当前 TODO 的下一项

T-075：acceptance-unit mapping 收敛。

具体第一步不是跑设备，而是创建可审核的 manifest/schema 草案，然后把 T-074 首批 5 个单元
逐个套入 schema，确认 contract 和 variants 是否合理。首批通过后，再审核 48 个
`no-test-found` 与 29 个 indirect acceptance units。

T-075 输出应包含：

- schema/字段定义；
- 首批 5 个单元的 contract/variant 决策；
- candidate → test → acceptance unit 多对多映射；
- needs-review 队列；
- denominator 是否可冻结的结论；
- 若数量变化，提供 v1 → v2 的逐项变更表。

## 7. 建议的下一条 Codex 任务

```text
执行 T-075，不运行大规模 GPU/NPU：基于 WORKFLOW.md 定义 acceptance-unit manifest/schema，
先人工复核 T-074 的 mm_plus_mm、pad mm/bmm/addmm、addmm 五个单元；以 PyTorch community
tests 为事实源，明确 contract、variants、tracking_mode、registration evidence 和 denominator
状态。保留 T-074 v1，不覆盖 CSV；输出首批 manifest、mapping review 和 v1→v2 变更建议。
```

## 8. 当前环境边界

- 所有新测试从 `/home/z50063656/tmp` 发起；
- 主环境为 `/home/z50063656/Pass/activate_pass.sh` 激活的 Conda `Pass`；
- 不在 PyTorch/torch_npu 源码树中 import `torch`；
- installed torch_npu wheel 与 `dist` 同名 wheel 哈希冲突仍未解除，不重装；
- T-055～T-073 的 Benchmark/isolated venv 结果保留原环境标签；
- 本轮只调整文档仓，不修改 PyTorch、torch_npu、Triton、runner、环境或 wheel。
