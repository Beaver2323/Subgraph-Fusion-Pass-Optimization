# 实验报告与数据索引

> 索引更新时间：2026-09-02 23:32 CST（UTC+08:00）
> 原则：报告保存当时环境和结论，不因主线变化回写历史；当前任务状态以
> `../docs/CURRENT_STATUS.md` 为准。

## 当前主线证据

| 文件 | 作用 | 当前边界 |
| --- | --- | --- |
| [t076_npu_completion_20260902.md](t076_npu_completion_20260902.md) | T-076 五个单元的 NPU/comparison 闭环及 addmm 运行态纠偏 | 正式闭环 5/5；1 个行为一致、4 个预期产品分歧 |
| [t076_pattern_gpu_npu_guide_20260902.md](t076_pattern_gpu_npu_guide_20260902.md) | T-076 每个 pattern/variant 的源码意图与 GPU/NPU 行为导读 | 20/20 variant 学习说明；P-018 候选单列 |
| [t077_gpu_preparation_20260902.md](t077_gpu_preparation_20260902.md) | 第二波 5 units / 11 direct cases / 17 variants 的人工映射、参数化入口与 GPU 交付 | 准备阶段历史；已被有效 reference supersede |
| [t077_gpu_reference_20260902.md](t077_gpu_reference_20260902.md) | T-077 GPU 文本 handoff 的 11/11 case、17/17 variant、环境与哈希复核 | reference 已冻结 |
| [t077_npu_completion_20260902.md](t077_npu_completion_20260902.md) | T-077 五单元 NPU/comparison 闭环与 MM lowering 修复验证 | 正式闭环 5/5；候选尚未合入 |
| [t077_pattern_gpu_npu_guide_20260902.md](t077_pattern_gpu_npu_guide_20260902.md) | T-077 pattern 意图、源码块、GPU/NPU 对照和修复代码 | 17/17 variant 已解释 |
| [REF-mm-plus-mm-native NPU 复现报告](../issues/REF-mm-plus-mm-native/复现报告.md) | 原生直接 `NO_TESTS`、最小 adapter、4/4 NPU 目标合同和 graph-mode 证据 | 统一 comparison 已落盘；`BEHAVIOR_UNCHANGED` |
| [REF-pad-mm-dynamic-m-native NPU 复现报告](../issues/REF-pad-mm-dynamic-m-native/复现报告.md) | 原生 `NO_TESTS`、TRITON-only lowering 阻断、产品 gate baseline | 归属单元已正式闭环 |
| [t076_gpu_reference_20260901.md](t076_gpu_reference_20260901.md) | 13/13 GPU direct valid、环境、FX signature 与结构化哈希 | reference 已冻结；原始大 artifacts 保留在 GPU |
| [t076_reference_runner_20260831.md](t076_reference_runner_20260831.md) | 首批 direct GPU/reference plan、runner、schema 与静态验证 | runner 已完成 GPU 执行；设计边界保留 |
| [t075_acceptance_unit_mapping_review_20260831.md](t075_acceptance_unit_mapping_review_20260831.md) | 首批 5 个 acceptance units 的 contract/variant 与证据角色人工复核 | 静态 mapping 完成；reference 已冻结 |
| [t074_upstream_pass_test_index_20260829.md](t074_upstream_pass_test_index_20260829.md) | T-074 registration candidate、community test 和 provisional unit 总结 | 静态 v1，不是冻结分母 |
| [candidate_test_index.csv](upstream_pass_test_index_20260829/candidate_test_index.csv) | 207 条 candidate/control 行与测试映射 | inventory 输入，动态状态均未运行 |
| [acceptance_units.csv](upstream_pass_test_index_20260829/acceptance_units.csv) | 188 个 heuristic 去重单元 | 158 eligible 仍为 provisional |
| [t056_triton_experimental_inventory_20260826.md](t056_triton_experimental_inventory_20260826.md) | T-074 的静态路由来源 | candidate discovery 辅助证据 |
| [triton_experimental_20260826/](triton_experimental_20260826/) | config、feature family 和 route CSV | previous inventory，不是任务分母 |

## 历史阶段导航

| 阶段 | 文件范围 | 说明 |
| --- | --- | --- |
| 初始 inventory/P0 | `pass_inventory*`、`pass_src_20260820/`、`p0_*` | default-backend 早期清单和基线 |
| MM/pad 第一批 | `t012_*`～`t025_t026_*` | mm_plus_mm、pad family 和替代方案证据 |
| torch_npu custom pass | `t027_*`～`t043_t046_*` | B2 结构、语义、alias 和性能 |
| DVM/MLIR/attention | `t047_*`～`t054_*` | B3/B4 历史结果 |
| experimental feature family | `t055_*`～`t072_*` | backend 启用、lowering、codegen、autotune 等专项证据 |

## 使用规则

1. 先从文件名日期和报告正文确认真实环境；
2. old Benchmark/isolated venv 结果不得自动升级为当前 Pass 环境 verdict；
3. 历史报告中的 “pass 数” 可能使用旧 inventory 口径，引用时必须同时说明是 registration、
   pattern family、custom pass 还是当前 acceptance unit；
4. 新 tracker 产物必须记录生成时间戳、schema version、source commit 和环境指纹；
5. 报告增长时优先更新本索引，不在根 README 重复堆叠完整实验叙述。
