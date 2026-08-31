# PyTorch Inductor Pass NPU 持续兼容性跟踪器

> 文档更新时间：2026-08-31 17:50 CST（UTC+08:00）
> 当前主线：PyTorch 社区原生 Inductor 优化契约在 NPU
> `triton_experimental` 后端上的持续兼容性验证。

本仓库从一次性审计仓升级为持续跟踪仓。它以 PyTorch upstream community tests
定义的行为为主要事实源，记录同一优化契约在 GPU/reference 与 NPU
`triton_experimental` 上的命中、正确性、运行路径、性能及版本变化。

远端仓库名称暂时保留 `Subgraph-Fusion-Pass-Optimization`，避免在框架和迁移路径尚未稳定时
破坏已有链接；逻辑定位已经切换为 compatibility tracker。仓库改名属于独立的远端管理动作，
不影响当前工作线。

## 当前结论

- 2026-08-29 之后的活动主线是 community-native Inductor compatibility tracker；
  default backend、torch_npu custom pass 和 T-055～T-073 feature-family 工作保留为历史证据。
- T-074 保存了 203 条 inherited-upstream registration candidate 和 4 条显式关闭控制项，
  共 207 行；这些是静态 inventory，不是 207 个 Pass。
- T-074 的自动规则暂时聚合出 188 个 acceptance unit，其中 158 个仅为
  `yes-provisional`。该聚合仍需按 upstream optimization contract 人工审核，不能作为冻结分母。
- 目前只有 5 个单元完成第一轮人工映射；尚未形成同时具备 GPU baseline、当前 Pass 环境 NPU
  结果和 comparison verdict 的正式闭环，因此新口径完成数仍为 0。
- 本阶段暂停新增大规模 pass 测例和 GPU/NPU 批量运行，先完成 T-074 映射收敛、manifest 和
  统一结果 schema。

## 统一术语

| 术语 | 本仓库含义 | 能否直接作为完成分母 |
| --- | --- | --- |
| registration site/candidate | 从源码静态识别的注册点、pipeline 入口或扩展 hook，用于 inventory 与查漏 | 不能 |
| pattern | PatternMatcher 实际注册和尝试匹配的模式；一个注册点可能展开多个 pattern | 不能自动等同 |
| community test | PyTorch upstream 测试及 helper，是 expected behavior 的主要事实源 | 是 acceptance unit 的证据，不单独等于单元 |
| acceptance unit | 一个需要独立回答“upstream optimization contract 在 NPU 上是否成立”的验证单元 | 人工审核冻结后可以 |
| variant | 同一 contract 下需要分别验证的 dtype、shape、layout、dynamic、forward/backward 等重要分支 | 由单元显式管理 |

严格禁止以下等式：

```text
registration == pattern == pass == acceptance unit
```

## 标准工作线

```text
upstream change / community test discovery
→ registration inventory 辅助 coverage 检查
→ acceptance-unit mapping
→ GPU/reference baseline
→ NPU triton_experimental execution
→ reference/NPU comparison
→ first-divergence failure classification
→ repair queue
→ regression tracking
```

GPU 与 NPU 位于不同机器。GPU 机器没有 Agent，只执行仓库生成的批量脚本并回传结构化
artifacts；NPU 机器负责映射、runner 生成、NPU 执行、差异分析、修复和回归。工作流不假设
同一个 Agent 同时控制两台机器。

## 文档入口

| 入口 | 用途 |
| --- | --- |
| [TODO.md](TODO.md) | 当前优先级、任务状态与完成标准 |
| [WORKFLOW.md](WORKFLOW.md) | 双机执行流程、schema、判定与修复状态机 |
| [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) | 2026-08-31 校准结论、T-074 边界与下一任务 |
| [docs/SCOPE_AND_CODE_MAP.md](docs/SCOPE_AND_CODE_MAP.md) | 任务范围、Inductor 调用链和源码入口 |
| [docs/GUIDE.md](docs/GUIDE.md) | 机制说明与历史案例阅读指南 |
| [docs/CHANGE_CONTROL.md](docs/CHANGE_CONTROL.md) | 环境、产品代码、文档和交付变更记录 |
| [docs/HISTORY.md](docs/HISTORY.md) | 已完成成果、失败与中性尝试索引 |
| [docs/requirements/20260831_requirement_change.md](docs/requirements/20260831_requirement_change.md) | 8 月 31 日需求合同与决策 |
| [docs/archive/README.md](docs/archive/README.md) | 旧检查点、旧计划、旧交接与旧总览 |
| [report/README.md](report/README.md) | 实验报告和 T-074 数据导航 |

## 当前下一步

下一条执行任务定义为 T-075：

1. 冻结 acceptance-unit manifest/schema，明确 `acceptance_unit_id`、contract、variant、
   community test、registration evidence 和 review status；
2. 用该 schema 复核 T-074 已人工映射的 5 个单元，判断是否需要合并或拆分；
3. 再审核 48 个 `no-test-found` 单元和 29 个 indirect 单元对应的 community test 证据；
4. 审核完成前不改变 188/158 的“provisional”性质，也不启动大规模设备实验。

T-075 收敛后进入 T-076：生成 manifest-driven GPU/reference runner 和人工操作说明。

## 执行环境合同

- 项目工作目录：`/home/z50063656/Pass`；
- 所有测试从 `/home/z50063656/tmp` 发起；
- 新测试只使用 `/home/z50063656/Pass/activate_pass.sh` 激活 Conda `Pass`；
- 不在 PyTorch 或 torch_npu 源码树中 import `torch`；
- 不使用旧 `Benchmark/env.sh` 启动新任务；
- installed wheel 与同名 `dist` wheel 的哈希冲突解决前，不得直接重装；
- 每个 experimental backend/pass-on/pass-off 对照使用 fresh process；
- correctness、fallback 和 graph-break 门禁通过前，不给出性能收益结论。

## 历史证据边界

历史报告不会因主线变化而删除或改写。旧 default backend、custom pass、feature-family、
Benchmark 和隔离 venv 结果都保留其原始环境标签，只能作为 previous-phase evidence；只有在
当前 manifest、reference baseline 和 NPU comparison 下重新关联后，才能升级为 tracker verdict。
