# Pass NPU 审计暂停检查点（2026-08-26，P-013/T-054）

> 历史检查点：本文件记录 2026-08-26 暂停时的 Benchmark 环境，不能作为 2026-08-28 之后
> 新测试的启动说明。当前入口是 `/home/z50063656/Pass/activate_pass.sh`；最新交接见
> [HANDOFF_20260828_PASS_ENV.md](../handoffs/HANDOFF_20260828_PASS_ENV.md)。

## 暂停状态

- 状态：`paused-by-user`。
- 已停止：新的 NPU correctness、benchmark、profiler、源码修改、wheel 构建与环境安装。
- 最后闭环：T-053/T-054，attention `_sfdp_pattern_5_half_inference` 的 NPU 性能负域与 P-013 精确 guard。
- 当前成果：旧 rewrite 功能正确但严重回退；P-013 已通过源码单测、源码 wheel、安装态 A/B、性能门禁和邻近 family 回归。
- 暂停时没有清理、终止或迁移其他用户进程；最后一次设备观察中物理 NPU 1 已出现外部任务，恢复时必须重新选择空闲卡。

## 固定环境

| 项目 | 暂停时状态 |
|---|---|
| 环境入口 | `/home/z50063656/Benchmark/env.sh` |
| 测试 cwd | 必须从 `/home/z50063656/tmp` 启动；不得在 torch_npu 源码目录内 import torch |
| Python | 3.11.15，Conda `benchmark-py311` |
| PyTorch | `2.14.0a0+git8e86e0a`，editable source 为 `/home/z50063656/Benchmark/pytorch-upstream`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0a0+git83cc452`，源码 commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc`，wheel 以 `--no-deps --force-reinstall` 安装 |
| Triton | import runtime `triton.__version__=3.2.0`；当前 pip metadata 为 `triton 3.5.0`、`triton_ascend 3.2.1`；审计源码 commit `8bd9f380d2786002b84b5248f00838c26f900515` |
| CANN / NPU | CANN 9.0.1；8 x Ascend 910B2 |

Triton 的 import runtime 版本与 pip metadata 不一致，是共享环境中已经存在、必须在后续独立
环境复验时消除的合同风险；本检查点同时记录两者，不把它静默改写成单一版本。`env.sh` 在当前
受限 shell 会打印一条 netlink 权限提示，但包元数据和 runtime 导入读取成功；本次归档没有启动
NPU 测试。

## P-013 归档与回滚边界

当前已安装、已验证 wheel 已从 `dist` 复制到稳定归档路径：

| 用途 | 文件 | SHA256 | 大小 |
|---|---|---|---:|
| 当前 P-013 | `artifacts/torch_npu_t054_p013_verified.whl` | `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989` | 33,770,835 B |
| 立即回滚到 P-012 | `artifacts/torch_npu_t053_before_p013.whl` | `61b0031cbb027548f60745dcf0a2484503a360347dec6bd3cc2f3f2bc823ebca` | 33,770,470 B |

P-013 的三个源码/测试文件另存于
`artifacts/p013_source_snapshot/`，文件级哈希和“累积源码快照”边界见该目录的
`MANIFEST.md`。其中两个产品 Python 文件与 P-013 wheel 内对应 entry 字节完全一致。

源码树不提交、不回退，符合“未创建单独环境时先用文档记录”的约束。P-013 只修改：

- `torch_npu/_inductor/fx_passes/joint_graph.py`：对 NPU default backend 的唯一 pattern-5 half-inference entry 加幂等 exact guard；
- `torch_npu/_inductor/__init__.py`：default backend 加载时安装 guard；
- `test/_inductor/test_attention_pattern_gate.py`：exact key、非 NPU、training、pattern 21 和重复安装合同。

没有修改 PyTorch 通用 pass、op-plugin、schema、C++ 或 Triton kernel。

## T-053 负向结论

pattern 5 输入为 `(4,8,128,64)` fp16 Q/K/V 和 `(1,1,128,128)` fp16 additive mask。
旧 rewrite 的 exact matcher 和 fusion counter 均命中，但 float mask 不能进入 vendor attention，
随后被 torch_npu dispatcher 重新展开：

| 指标（三轮中位） | 保留原图 | 旧 rewrite | 变化 |
|---|---:|---:|---:|
| P50 | 0.381385 ms | 0.775105 ms | 回退 103.23% |
| P99 | 0.409750 ms | 0.824400 ms | 回退 101.20% |
| 首次 compile+run | 41,807.41 ms | 100,682.28 ms | 回退 140.82% |
| device task/step | 3 | 8 | 增加 5 个 task |
| additional allocated peak | 204,472,832 B | 205,527,552 B | 增加 1,054,720 B |

两侧最大绝对误差均为 `0.0029296875`，并在 `atol=rtol=0.02` 下通过。这里的结论是
“数值满足约定，但图重写性能失败”，不是算子不可用，也不能由“误差很小”推出语义边界完整。

## T-054 P-013 正向结论

同一个新 wheel 内恢复旧 generator 作为 B 侧、默认 guard 作为 C 侧，三轮 fresh-process
paired、warmup 10、runs 100：

| 指标（三轮中位） | 旧 rewrite | P-013 guard | 改善 |
|---|---:|---:|---:|
| P50 | 0.745200 ms | 0.370545 ms | 50.28% |
| P99 | 0.767770 ms | 0.581510 ms | 24.26% |
| mean | 0.746317 ms | 0.382204 ms | 48.79% |
| 首次 compile+run | 98,470.86 ms | 41,546.24 ms | 57.81% |
| device task/step | 8 | 3 | 减少 62.5% |
| additional allocated peak | 205,527,552 B | 204,472,832 B | 减少 1,054,720 B |

所有预登记 gate 通过。pattern 1、13、21 fresh neighbor regression 均 exact/总 counter `1/1`
且数值通过；1/13 保持 vendor attention，21 保持 math fallback。guard 方案为
`supported-beneficial`；被 guard 的 rewrite 在矩阵中最终记
`supported-pass-disabled-performance-rejected`。

## 当前矩阵与里程碑

当前矩阵为 251 行、31 列：

| verdict | 数量 |
|---|---:|
| `not-run` | 220 |
| `supported-beneficial` | 9 |
| `supported-neutral` | 9 |
| `unsupported` | 4 |
| `supported-pass-disabled-performance-rejected` | 3 |
| `supported-neutral-resource-beneficial` | 3 |
| `not-applicable` | 2 |
| `conditional-supported-beneficial` | 1 |

已完成 P0，P1 B2 的 27 个 custom pass，B3 的 8 个 DVM/MLIR 记录，以及 B4 八个代表
attention family 的功能路径。B4 已关闭 pattern 1、13、5 的性能结论；成功、性能失败、中性、
reachability 和环境失败均已写入报告与 `outcome_index.md`，不能只统计“跑通”。

## 暂停时工作树

- torch_npu base commit 不变，tracked tree 保留 T-011、T-023、B2、P-012、P-013 等已登记的累积修改；P-013 三个直接文件见上节。大量 C++/dispatch 未跟踪文件属于构建代码生成产物，不纳入 P-013 功能变更，也不清理。
- editable PyTorch 有本任务开始前/共享环境中的 tracked 修改；P-013 未改 PyTorch。
- Triton Ascend 保留进入任务前已有的三个 tracked 修改；P-013 未改 Triton。
- 公共文档仓库在本次归档前为 clean，远端 `main` 最新提交为 `e5f0969`（`docs: record neutral T054 attempts`）。归档提交完成后以新的文档 commit 为准。

不要用整个 dirty worktree 直接声称“P-013 diff”。精确恢复应使用当前/回滚 wheel、P-013 三文件
snapshot 和 `change_control.md` 的 P-013 条目。

## 精确恢复点

恢复工作时先阅读本文、`current_status_and_background.md`、`outcome_index.md`、T-053/T-054
报告和 `change_control.md` 的 P-013/T-054 记录，然后：

1. 只读复核源码 commit、当前安装包、wheel SHA256 与三个工作树；不要重新构建或安装，除非哈希/导入路径不一致。
2. 运行 `npu-smi info`，只选择当时进程表明确为空的物理卡；不得沿用本检查点记录的历史卡号。
3. 不重复 T-053/T-054。下一主任务登记为 T-055：先对 pattern 21 做与 pattern 5 同口径的 exact pass-on/off paired，随后独立评估 pattern 29。
4. pattern 21/29 必须分别判断 matcher、最终 codegen、正确性、P50/P99、首次编译、task 和峰值内存；不能从 pattern 5 外推。
5. 只有当原 rewrite 不可用或明显变慢、且已有高概率单 task 替代方案时，才预登记手写 Triton 微原型；任意 float additive mask 不得直接转 bool。
6. 独立环境支线只保留 T-023 的无 audit shim fresh-launcher 复验，以及补齐 `torch_mlir` 后的完整 MLIR 复验。

## 证据入口

- [T-053 pattern 5 性能负结论](../../../report/t053_b4_attention_pattern5_performance_20260826.md)
- [T-054 P-013 guard 闭环](../../../report/t054_b4_attention_pattern5_guard_20260826.md)
- [当前状态与背景](../../CURRENT_STATUS.md)
- [成功/失败/中性结果索引](../../HISTORY.md)
- [变更控制记录](../../CHANGE_CONTROL.md)
- [入门指南](../../GUIDE.md)
- `results/t053_b4_attention_pattern5_performance_20260826/aggregate/aggregate.json`
- `results/t054_b4_attention_pattern5_guard_performance_20260826/aggregate/aggregate.json`
