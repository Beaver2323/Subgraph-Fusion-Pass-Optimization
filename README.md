# PyTorch Inductor Pass NPU 持续兼容性跟踪器

> 文档更新时间：2026-09-06 06:43 CST（UTC+08:00）
> 当前主线：PyTorch 社区原生 Inductor 优化契约在 NPU
> `triton_experimental` 后端上的持续兼容性验证。

本仓库从一次性审计仓升级为持续跟踪仓。它以 PyTorch upstream community tests
定义的行为为主要事实源，记录同一优化契约在 GPU/reference 与 NPU
`triton_experimental` 上的命中、正确性、运行路径、性能及版本变化。

远端仓库名称暂时保留 `Subgraph-Fusion-Pass-Optimization`，避免在框架和迁移路径尚未稳定时
破坏已有链接；逻辑定位已经切换为 compatibility tracker。仓库改名属于独立的远端管理动作，
不影响当前工作线。

## 当前结论

- 新规则复核：T-076/T-077 的 10 份 NPU/comparison 记录通过；原始 GPU/NPU/性能证据再认证仍为
  `pending`。下列历史闭环与收益不等于已完成新规则全量重验。详见
  [逐单元清单与统一门禁](report/t076_t077_history_reaudit_20260906.md)，机器可读入口为
  [最新审计](results/audits/latest.json)。
- 2026-08-29 之后的活动主线是 community-native Inductor compatibility tracker；
  default backend、torch_npu custom pass 和 T-055～T-073 feature-family 工作保留为历史证据。
- T-074 保存了 203 条 inherited-upstream registration candidate 和 4 条显式关闭控制项，
  共 207 行；这些是静态 inventory，不是 207 个 Pass。
- T-074 的自动规则暂时聚合出 188 个 acceptance unit，其中 158 个仅为
  `yes-provisional`。该聚合仍需按 upstream optimization contract 人工审核，不能作为冻结分母。
- T-075 已把首批 5 个单元写入 schema/manifest，并完成 contract/variant 人工复核；
  T-076 已完成全部 5 个单元的 GPU baseline、NPU 结果与 comparison，正式闭环 `5/5`。
- T-076 的 GPU 环境与精确 source build 已验真，13/13 direct community cases 均 passed 且
  `reference_valid=true`；不存在 adapter case。GPU 禁止 Git/二进制上传，已通过通用文本导出器
  回传并复核环境、summary、逐 case FX signature 和关键文件哈希；对应 NPU comparison 已完成。
- `AU-post-grad-mm-plus-mm` 已按原生优先执行：直接入口因上游 `HAS_GPU`
  不包含 NPU 而为 `NO_TESTS`；case-specific adapter 在 `triton_experimental` 上
  4/4 输入分支有效，统一 NPU/comparison 记录已落盘，正式 verdict 为
  `BEHAVIOR_UNCHANGED`。
- `AU-pad-mm-mm` 的 dynamic-M、original-aten、stride、exclusion 四个 case 均完成原生优先与
  case-specific adapter；产品 gate 始终保持 `disable_pad_mm=true`，原图 correctness/stride
  有效，正式 verdict 为 `EXPECTED_PRODUCT_DIVERGENCE`。
- `AU-pad-mm-bmm` 和 `AU-pad-mm-addmm` 已在不绕过产品 gate 的前提下闭环，verdict 均为
  `EXPECTED_PRODUCT_DIVERGENCE`。
- `AU-post-grad-addmm` 已完成运行态纠偏：当前 Pass 安装态明确
  `disable_addmm_fusion=True`，因此 GPU `2/4` 与安装态 NPU `0/0` 是
  `EXPECTED_PRODUCT_DIVERGENCE`；P-018 独立候选已恢复正例 `2/4` 并保留全部负例 `0/0`。
- T-077 第二波 GPU reference 11/11 direct cases、17/17 variants 有效，NPU 5/5 单元正式闭环。
  Gumbel 功能为 `BEHAVIOR_UNCHANGED`，同 backend 性能最终为 `PERF_IMPROVED`；B2B GEMM、decompose-BMM、dynamic addmm 为
  `EXPECTED_PRODUCT_DIVERGENCE`；decompose-MM 发现 small-mm pointwise 反向 lowering 回归，
  本地候选 `dfbcc25` 已通过 2/2 定向单测和 6/6 合同回归，尚未推送/合入。
- T-076 性能处置为 2 个 experimental 实测和 3 个显式关闭免测。T-077 性能处置也已 5/5：
  Gumbel 为 `PERF_IMPROVED`，B2B 为 `CAPABILITY_REJECTED_NO_EFFECTIVE_TEMPLATE`，三个
  decompose 最小候选均为 `PERF_REGRESSED` 并保留 NPU guard。default backend 历史结果不得迁移为
  experimental verdict。
- T-078 已从 no-test-found/indirect inventory 人工复核出 4 个新 acceptance units：addcdiv→FMA、
  partial reduction reuse、addmm bias unfuse、baddbmm bias unfuse；已收到 12/12 通过的 GPU 1.0
  紧凑摘要，但该格式不含 FX/日志正文，尚待用 1.1 原文 handoff 补齐复核，不计入冻结 denominator。
- T-079/T-080 又准备 7 个 acceptance units、17 个 direct cases、27 个 variants：T-079 覆盖
  bmm→mm 与三类 cat/split lowering；T-080 覆盖 const-scatter、prepare-softmax 和 constructor
  mover。连同 T-078，11 个单元均已补齐性能计划和中文功能/性能测例讲解；计划区分社区 benchmark
  直接复用与社区功能例派生，并冻结 `triton_experimental`、fresh-process、目标级 OFF/ON 和门禁。
  三批仍等待 GPU reference，不计入冻结 denominator，也尚无性能结论。
- T-078 的 `copy_tests` 入口已纠偏：社区实际方法带 `_cuda` 后缀；runner 不再把不存在的无后缀
  方法静态判为有效。

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
| [docs/TASK_BACKLOG.md](docs/TASK_BACKLOG.md) | T-081～T-113 草案批次；完整单元列表、非计数记录及 lowering/template 覆盖边界 |
| [report/tracker_validation_hardening_20260906.md](report/tracker_validation_hardening_20260906.md) | 验收误判修复、零设备回归与性能准备尚未完成的边界 |
| [docs/GUIDE.md](docs/GUIDE.md) | 机制说明与历史案例阅读指南 |
| [docs/REFERENCE_RUNNER_GPU.md](docs/REFERENCE_RUNNER_GPU.md) | T-076 GPU 静态校验、整批执行、重跑、二进制/文本回传说明 |
| [docs/T077_REFERENCE_RUNNER_GPU.md](docs/T077_REFERENCE_RUNNER_GPU.md) | T-077 第二波 GPU 执行与文本回传说明 |
| [docs/T078_REFERENCE_RUNNER_GPU.md](docs/T078_REFERENCE_RUNNER_GPU.md) | T-078 第三批 GPU 执行、case 列表与文本回传说明 |
| [docs/T079_REFERENCE_RUNNER_GPU.md](docs/T079_REFERENCE_RUNNER_GPU.md) | T-079 第四批 GPU 执行、case 列表与文本回传说明 |
| [docs/T080_REFERENCE_RUNNER_GPU.md](docs/T080_REFERENCE_RUNNER_GPU.md) | T-080 第五批 GPU 执行、社区性能来源与文本回传说明 |
| [docs/GPU_TASK_RUNNER.md](docs/GPU_TASK_RUNNER.md) | GPU 一键运行（默认共享、可选独占、快速等卡）、固定 latest 与 results/incoming/ JSON 接收路径 |
| [docs/GPU_TEXT_HANDOFF.md](docs/GPU_TEXT_HANDOFF.md) | 1.0/1.1 格式边界、原文导出、GitHub 文本复制、校验、安全恢复与 FX 查看 |
| [results/incoming/README.md](results/incoming/README.md) | GPU 文本 handoff 接收目录，T-076～T-080 文件夹随 clone/pull 建立 |
| [docs/T078_FUNCTION_PERFORMANCE_GUIDE.md](docs/T078_FUNCTION_PERFORMANCE_GUIDE.md) | T-078 四单元的功能 case、派生性能 case、源码与判据讲解 |
| [docs/T079_FUNCTION_PERFORMANCE_GUIDE.md](docs/T079_FUNCTION_PERFORMANCE_GUIDE.md) | T-079 四单元的图消除功能/性能证据讲解 |
| [docs/T080_FUNCTION_PERFORMANCE_GUIDE.md](docs/T080_FUNCTION_PERFORMANCE_GUIDE.md) | T-080 社区 benchmark 复用、功能 guard 与 OFF/ON 讲解 |
| [report/t076_t077_performance_20260903.md](report/t076_t077_performance_20260903.md) | 两批性能处置、backend 门禁、B2B capability 与 T-077 四项 OFF/ON 实测 |
| [report/t076_pattern_gpu_npu_guide_20260902.md](report/t076_pattern_gpu_npu_guide_20260902.md) | T-076 20 个 variants 的源码意图、GPU/NPU 行为与 P-018 候选导读 |
| [report/t077_npu_completion_20260902.md](report/t077_npu_completion_20260902.md) | T-077 五单元闭环、MM 回归定位与修复验证摘要 |
| [report/t077_pattern_gpu_npu_guide_20260902.md](report/t077_pattern_gpu_npu_guide_20260902.md) | T-077 每个 pattern/variant 的源码意图、GPU/NPU 行为和修复代码导读 |
| [report/t078_mapping_review_20260903.md](report/t078_mapping_review_20260903.md) | T-078 四单元映射修正、源码意图、12 cases/20 variants 与 GPU 前置合同 |
| [report/t079_t080_mapping_review_20260904.md](report/t079_t080_mapping_review_20260904.md) | T-079/T-080 七单元的源码意图、映射修正、性能来源和后端边界 |
| [docs/CHANGE_CONTROL.md](docs/CHANGE_CONTROL.md) | 环境、产品代码、文档和交付变更记录 |
| [docs/HISTORY.md](docs/HISTORY.md) | 已完成成果、失败与中性尝试索引 |
| [docs/requirements/20260831_requirement_change.md](docs/requirements/20260831_requirement_change.md) | 8 月 31 日需求合同与决策 |
| [docs/archive/README.md](docs/archive/README.md) | 旧检查点、旧计划、旧交接与旧总览 |
| [upstream/README.md](upstream/README.md) | Acceptance-unit schema、manifest 与多对多映射入口 |
| [report/README.md](report/README.md) | 实验报告、T-074 数据和 T-075 复核导航 |

## 当前下一步

T-075 首批静态复核已完成：

1. 首批仍为 5 个 acceptance units，共 20 个 variants、13 个 community test 引用；
2. `mm_plus_mm` 的 same-K/different-K 保持一个 contract；pad mm/bmm/addmm 保持三个；
   两种 add+mm 顺序共享一个 addmm contract；
3. GPU 文本证据已复核，5 个单元冻结进入 denominator；T-076 已完成统一 comparison，正式闭环为 5/5；
4. 48 个 `no-test-found` 和 29 个 indirect 单元继续待人工审核，T-074 v1 不覆盖。

T-076 GPU reference 已完成：13 个原生 community cases 全部 direct valid，20 个 variants 中 14 个取得动态
reference，3 个 registration-only 和 3 个 NPU-only gate 保持显式非动态处置。完整环境、逐 case
FX signature 与结果/inventory 哈希见 `report/t076_gpu_reference_20260901.md`。首个 NPU 单元证据见
`issues/REF-mm-plus-mm-native/复现报告.md` 和
`results/current/REF-mm-plus-mm-native/`。pad-mm 单元级 comparison 见
`results/current/AU-pad-mm-mm/`。T-076 NPU 闭环汇总见
`report/t076_npu_completion_20260902.md`。T-077 第二波 reference 和 NPU 5/5 comparison 也已完成；
GPU 后续统一使用 `scripts/run_gpu_reference_task.sh --task T-076|T-077|T-078|T-079|T-080 --gpu ID`。
T-077 性能 5/5 已处置、pending=0；T-078/T-079/T-080 均已完成 reference、逐单元性能合同和
中文 case guide 的静态准备，GPU 可按任务分别一键执行。优先完成 T-078，再依次回传 T-079 与
T-080 的 latest 文本 handoff。三批 NPU 性能目前仅有方案，worker 尚未实现；必须先完成实现和
静态验证，再经 reference 与 NPU 功能/命中门禁开放实测，不会自动解锁。
独立 correctness 修复 `dfbcc25` 继续等待产品代码评审授权，不与 T-078 reference 混合。

2026-09-06 验收加固：部分 skip、expected failure、执行数不足均不算 valid reference；NPU
结果强制校验 `triton_experimental`；未修复数值回归可作为合法失败证据落盘，但不算正式闭环。
GPU 导出失败时 `latest-text-handoff.json` 显示本轮失败状态，不再保留上轮成功入口。
后续 137 个 provisional eligible 单元已暂列 T-081～T-113，共 33 个**待审核草案**，不是可运行批次；
另有 30 条非计数结构记录。详见上方 backlog，不将其计入冻结分母。

## 执行环境合同

- 项目工作目录：`/home/z50063656/Pass`；
- NPU 新测试从 `/home/z50063656/tmp` 发起，并使用
  `/home/z50063656/Pass/activate_pass.sh` 激活 Conda `Pass`；
- GPU T-076 测试从 `/data/z50063656/tmp` 发起，使用
  `/data/z50063656/envs/PassGPURef` pip venv，重型环境、源码、缓存和产物均写入 `/data`；
- 不在 PyTorch 或 torch_npu 源码树中 import `torch`；
- 不使用旧 `Benchmark/env.sh` 启动新任务；
- installed wheel 与同名 `dist` wheel 的哈希冲突解决前，不得直接重装；
- 每个 experimental backend/pass-on/pass-off 对照使用 fresh process；
- correctness、fallback 和 graph-break 门禁通过前，不给出性能收益结论。

## 历史证据边界

历史报告不会因主线变化而删除或改写。旧 default backend、custom pass、feature-family、
Benchmark 和隔离 venv 结果都保留其原始环境标签，只能作为 previous-phase evidence；只有在
当前 manifest、reference baseline 和 NPU comparison 下重新关联后，才能升级为 tracker verdict。
