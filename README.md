# Subgraph Fusion Pass Optimization

面向昇腾 NPU 的 PyTorch Inductor pass 可用性、正确性和性能审计项目。目标是逐项确认
Inductor pass 在 NPU 上是否真实触发、能否正确 lowering/codegen、是否带来稳定收益，
并只在证据充分时评估 NPU lowering、vendor op、AscendC 或手写 Triton 替代。

## 从这里开始

1. [从头学习指南](inductor_pass_npu_beginner_guide.md)：编译链、背景知识、源码入口和实验方法。
2. [当前状态与环境](current_status_and_background.md)：运行环境、已完成结论和当前进行位置。
3. [成果、失败与中性索引](outcome_index.md)：快速区分有效成果、否决方案、未归因和环境阻塞。
4. [项目审计总览](audit_overview.md)：清单、P0/P1 工作流和报告导航。
5. [任务范围与源码地图](task_scope_and_code_map.md)：需要阅读、修改和验证的源码区域。
6. [变更控制记录](change_control.md)：每次实验、修改范围、回滚方法和证据日志。
7. [替代与优化计划](replacement_plan.md)：何时保留 pass、修 gate、复用 vendor op 或考虑 Triton。

## 当前进度

- 已建立 251 条 pass 评估矩阵。
- `addmm fusion`：`supported-beneficial`。
- different-K `mm_plus_mm`：已完成 default-off NPU template wheel 集成，首批 p50
  提升 15.29%/18.04%，但存在显存和正式 no-shim 环境条件，当前为
  `conditional-supported-beneficial`。
- `pad_mm/pad_bmm/pad_addmm`：测试侧绕过 device gate 后功能正确，但 p50 分别回退
  72.65%/65.31%/120.63%，因此保持产品 gate，不实现独立 Triton padding 替身。
- P1 B2 前四批：FX 测试 60/60 通过。`fold_reduce` 的直返输入存在 alias 错误，clone
  虽修复正确性但 p50/p99 回退 3.06%/6.72%，最终 wheel 保留原 sum并禁用折叠。
  `cat_to_view_pass` 使用 alias-safe clone 后 p50 +2.29%、task 3→1、额外 allocated
  peak 减少 4,195,840 B，当前为 `supported-neutral-resource-beneficial`。
  `fold_cat` 的 nested cat `2→1`，p50/p99 改善 10.14%/10.32%、task 2→1、额外
  allocated peak 减少 2,097,664 B，当前为 `supported-beneficial`。
  第三批另验证 5 条 view/copy/where pass；`fold_where` 功能正确但 p50/p99 仅改善
  1.16%/3.12%、task 和显存不变，当前为 `supported-neutral`。
  第四批发现 `cat_slice_cat_fold_pass` 与 `pad_slice_fold` 在数值误差为 0 时仍破坏
  alias/stride，现已用保守 guard 修复；三轮 paired p50 分别改善 24.00%/31.35%，
  task 2→1/3→1，两条均为 `supported-beneficial`。
- 矩阵当前为 240 条 `not-run`，另有 11 条最终、条件性或中性 verdict；下一步是 B2
  其余 9 个 custom pass，再进入 DVM/MLIR 和 attention。

机器可读矩阵位于
[pass_evaluation_matrix.csv](report/pass_src_20260820/pass_evaluation_matrix.csv)，填写规则和
汇总见 [pass_evaluation_matrix.md](report/pass_src_20260820/pass_evaluation_matrix.md)。

## 重点报告

- [addmm 与 P0 语义矩阵](report/p0_semantic_matrix_20260821.md)
- [different-K mm_plus_mm 集成](report/t023_mmplus_different_k_integration_20260821.md)
- [different-K workspace 审计](report/t024_mmplus_different_k_workspace_20260821.md)
- [pad family 功能与性能结论](report/t025_t026_pad_family_20260821.md)
- [P1 B2 首批结构验收](report/t027_p1_b2_structure_20260821.md)
- [P1 B2 NPU compile 与 alias 闭环](report/t028_p1_b2_npu_compile_20260821.md)
- [B2 alias 修复、性能与最终 wheel](report/t029_t030_b2_alias_fix_performance_20260824.md)
- [B2 第二批结构与 NPU 编译](report/t032_b2_redundancy_compile_20260824.md)
- [fold_cat 单 pass NPU 性能](report/t033_fold_cat_performance_20260824.md)
- [B2 第三批结构与 NPU 编译](report/t034_b2_view_copy_compile_20260824.md)
- [fold_where 单 pass NPU 性能](report/t035_fold_where_performance_20260824.md)
- [B2 第四批 alias 修复与源码 wheel](report/t036_b2_layout_alias_fix_20260825.md)
- [cat-slice-cat / pad-slice 单 pass NPU 性能](report/t037_layout_pass_performance_20260825.md)

## 当前验证环境

- Python 3.11.15
- PyTorch `2.14.0a0+git8e86e0a` editable source
- torch_npu `2.14.0a0+git83cc452` source-built wheel，使用 `--no-deps` 安装
- Triton runtime module 3.2.0
- CANN 9.0.1
- 8 × Ascend910B2

当前安装的 T-036 torch_npu wheel SHA256 为
`d745cf3afd6a2859a68d6c31dd02a46498264e82dedff34d726c2be2609c6b9d`；
T-031 旧 wheel 已在本地审计目录保留用于回滚，不上传仓库。

仓库当前保存文档、报告和小型机器可读清单；不包含 wheel、编译缓存、profiler 原始数据、
设备运行缓存或任何访问凭据。实验命令应从 `/home/z50063656/tmp` 启动，避免在
torch_npu 源码目录内导入 `torch`。
