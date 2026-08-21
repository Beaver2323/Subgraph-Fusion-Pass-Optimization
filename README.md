# Subgraph Fusion Pass Optimization

面向昇腾 NPU 的 PyTorch Inductor pass 可用性、正确性和性能审计项目。目标是逐项确认
Inductor pass 在 NPU 上是否真实触发、能否正确 lowering/codegen、是否带来稳定收益，
并只在证据充分时评估 NPU lowering、vendor op、AscendC 或手写 Triton 替代。

## 从这里开始

1. [从头学习指南](inductor_pass_npu_beginner_guide.md)：编译链、背景知识、源码入口和实验方法。
2. [当前状态与环境](current_status_and_background.md)：运行环境、已完成结论和当前进行位置。
3. [项目审计总览](audit_overview.md)：清单、P0/P1 工作流和报告导航。
4. [任务范围与源码地图](task_scope_and_code_map.md)：需要阅读、修改和验证的源码区域。
5. [变更控制记录](change_control.md)：每次实验、修改范围、回滚方法和证据日志。
6. [替代与优化计划](replacement_plan.md)：何时保留 pass、修 gate、复用 vendor op 或考虑 Triton。

## 当前进度

- 已建立 251 条 pass 评估矩阵。
- `addmm fusion`：`supported-beneficial`。
- different-K `mm_plus_mm`：已完成 default-off NPU template wheel 集成，首批 p50
  提升 15.29%/18.04%，但存在显存和正式 no-shim 环境条件，当前为
  `conditional-supported-beneficial`。
- `pad_mm/pad_bmm/pad_addmm`：测试侧绕过 device gate 后功能正确，但 p50 分别回退
  72.65%/65.31%/120.63%，因此保持产品 gate，不实现独立 Triton padding 替身。
- P1 B2：7 个 custom pass 的 FX 结构测试 32/32 通过；`fold_reduce` 已完成 default
  backend NPU positive/negative 功能闭环，其他 pass 和性能仍在进行。

机器可读矩阵位于
[pass_evaluation_matrix.csv](report/pass_src_20260820/pass_evaluation_matrix.csv)，填写规则和
汇总见 [pass_evaluation_matrix.md](report/pass_src_20260820/pass_evaluation_matrix.md)。

## 重点报告

- [addmm 与 P0 语义矩阵](report/p0_semantic_matrix_20260821.md)
- [different-K mm_plus_mm 集成](report/t023_mmplus_different_k_integration_20260821.md)
- [different-K workspace 审计](report/t024_mmplus_different_k_workspace_20260821.md)
- [pad family 功能与性能结论](report/t025_t026_pad_family_20260821.md)
- [P1 B2 首批结构验收](report/t027_p1_b2_structure_20260821.md)
- [P1 B2 NPU compile（进行中）](report/t028_p1_b2_npu_compile_20260821.md)

## 当前验证环境

- Python 3.11.15
- PyTorch `2.14.0a0+git8e86e0a` editable source
- torch_npu `2.14.0a0+git83cc452` source-built wheel，使用 `--no-deps` 安装
- Triton runtime module 3.2.0
- CANN 9.0.1
- 8 × Ascend910B2

仓库当前保存文档、报告和小型机器可读清单；不包含 wheel、编译缓存、profiler 原始数据、
设备运行缓存或任何访问凭据。实验命令应从 `/home/z50063656/tmp` 启动，避免在
torch_npu 源码目录内导入 `torch`。
