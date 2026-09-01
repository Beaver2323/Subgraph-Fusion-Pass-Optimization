# PyTorch Inductor Pass NPU 持续兼容性跟踪器

> 文档更新时间：2026-09-02 05:38 CST（UTC+08:00）
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
- T-075 已把首批 5 个单元写入 schema/manifest，并完成 contract/variant 人工复核；
  T-076 已完成全部 5 个单元的 GPU baseline、NPU 结果与 comparison，正式闭环 `5/5`。
- T-076 的 GPU 环境与精确 source build 已验真，13/13 direct community cases 均 passed 且
  `reference_valid=true`；不存在 adapter case。GPU 禁止 Git/二进制上传，已通过通用文本导出器
  回传并复核环境、summary、逐 case FX signature 和关键文件哈希，当前进入 NPU comparison。
- `AU-post-grad-mm-plus-mm` 已按原生优先执行：直接入口因上游 `HAS_GPU`
  不包含 NPU 而为 `NO_TESTS`；case-specific adapter 在 `triton_experimental` 上
  4/4 输入分支有效，统一 NPU/comparison 记录已落盘，正式 verdict 为
  `BEHAVIOR_UNCHANGED`。
- `AU-pad-mm-mm` 的 dynamic-M、original-aten、stride、exclusion 四个 case 均完成原生优先与
  case-specific adapter；产品 gate 始终保持 `disable_pad_mm=true`，原图 correctness/stride
  有效，正式 verdict 为 `EXPECTED_PRODUCT_DIVERGENCE`。
- `AU-pad-mm-bmm` 和 `AU-pad-mm-addmm` 已在不绕过产品 gate 的前提下闭环，verdict 均为
  `EXPECTED_PRODUCT_DIVERGENCE`。
- `AU-post-grad-addmm` 的负向 guard 正确，但 matrix/vector bias 正例 target counter 从 GPU `2/4`
  变为 NPU `0/0`，verdict 为 `NPU_REGRESSION`，已进入 repair queue。
- T-077 第二波 5 个待 reference 单元已完成人工映射：11 个 direct cases、17 个 variants，
  零设备校验通过，等待 GPU 执行；它们尚未进入冻结 denominator。

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
| [docs/REFERENCE_RUNNER_GPU.md](docs/REFERENCE_RUNNER_GPU.md) | T-076 GPU 静态校验、整批执行、重跑、二进制/文本回传说明 |
| [docs/T077_REFERENCE_RUNNER_GPU.md](docs/T077_REFERENCE_RUNNER_GPU.md) | T-077 第二波 GPU 执行与文本回传说明 |
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
`report/t076_npu_completion_20260902.md`；GPU 侧下一步按
`scripts/run_t077_reference_all.sh` 执行第二波 reference。

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
