# P0 语义与 Forward/Backward 矩阵（2026-08-21）

## 当前结论

addmm/default 和 mm_plus_mm/triton_experimental 的新增 inference 语义矩阵 6/6 `compile-correct`，正例与负例的 generated code 均符合源码 guard。初始两个不含 broadcast reduction 的 backward 隔离用例通过；T-011 修复 reduction `strict_sum` 接口后，原 vector-bias backward blocker 和全部近邻回归也通过。

原始唯一未通过项是 addmm vector-bias backward：forward 已融合为 `aten.addmm`，失败发生在 bias 梯度 `aten.sum.dim_IntList` lowering，错误为 `make_reduction() got an unexpected keyword argument 'strict_sum'`。T-011 在 torch_npu 覆盖函数中补齐 keyword-only 参数并透传 `Reduction.create()` 后，vector-bias case 为 `compile-correct`：forward 最大误差 0.03125（fp16 容差内），三个输入梯度最大误差均为 0。因此 addmm 最终 verdict 已升级为 `supported-beneficial`。

## 语义结果

| family/backend | case | 结果 | 目标图或 fallback | 解释 |
|---|---|---|---|---|
| addmm/default | `(M,N)` full bias | correct，最大误差 0 | 单个 `aten.addmm` | 正例触发 |
| addmm/default | `(1,N)` row bias | correct，最大误差 0.03125，fp16 容差通过 | 单个 `aten.addmm` | broadcast 正例触发 |
| addmm/default | fp16 mm + fp32 bias | correct，最大误差 0 | `extern_kernels.mm` + add，无 `aten.addmm` | dtype guard 正确拒绝 |
| mm_plus_mm/experimental | 相同 M/N、不同 K | correct，最大误差 0 | post-grad 有 marker；lowering 为两个 mm + Triton add | pattern 匹配后安全 unfuse |
| mm_plus_mm/experimental | M 维 broadcast | correct，最大误差 0 | 两个 mm + add，无 marker | shape guard 正确拒绝 |
| mm_plus_mm/experimental | N 维 broadcast | correct，最大误差 0 | 两个 mm + add，无 marker | shape guard 正确拒绝 |

三个 addmm inference worker 和三个 mm_plus_mm inference worker 均使用 fp16、small `(M,K,N)=(32,64,48)`、contiguous、static、current 模式，不采性能。dtype 负例的 bias 固定为 fp32；其输出为 fp32。

## Forward/Backward 结果

| family/backend | bias/shape | forward | 梯度 | 结论 |
|---|---|---|---|---|
| mm_plus_mm/experimental | 两条相同 K 的 `(32,64)@(64,48)` | `_mm_plus_mm` extern | 4 个输入梯度全部正确，最大误差 0 | pass 的 AOTAutograd 路径可用 |
| addmm/default | `(M,N)` full bias | `aten.addmm` | 3 个输入梯度全部正确，最大误差 0 | addmm pass 本身的 forward/backward 可用 |
| addmm/default | `(N,)` vector bias | `aten.addmm` | 修复前被 reduction 接口阻断；T-011 后 3 个输入梯度最大误差均为 0 | 通用 blocker 已关闭 |

backward 探针让 eager 与 compiled 使用同值但独立的叶子 tensor，分别执行标量 loss backward，然后逐个比较输出和输入梯度。初始两个通过用例的首次 compiled forward + backward 分别约为 13.772 s 和 15.895 s；T-011 后 vector-bias fresh-worker 首次编译约 32.760 s。这些值包含编译且 cache 策略不同，只用于诊断，不是性能结论。

修复前 vector-bias 失败的 backward FX 图包含：

```text
aten.sum.dim_IntList(tangents_1, [0], True)
aten.reshape(..., [48])
```

因此原失败与 bias 的 broadcast 梯度归约直接相关。full-bias 不需要这一步，所以能隔离验证 addmm pass 和 matmul 梯度本身；T-011 后同一 vector-bias case 已跨过该 lowering 并完成梯度比较。

## 源码解释

addmm 的 pattern 条件来自 `torch/_inductor/fx_passes/post_grad.py:1983`：bias 必须可扩展到 `(M,N)`，且 bias、mat1、mat2 dtype 相同。动态结果与它一致：full/row bias 触发，mixed dtype 不触发。

mm_plus_mm 的 pattern 条件来自 `post_grad.py:979`：分别验证两条 matmul 的内部 K，要求两个输出的 M/N 相同，但不要求两条 matmul 的 K 相同。因此不同 K case 会出现 post-grad marker。

真正的 fusion lowering 在 `torch/_inductor/kernel/mm_plus_mm.py:128`。当前实现要求 mat1 与 mat3、mat2 与 mat4 的 size 分别相同；否则调用两个 mm 和 add 的 lowerings。不同 K case 的 generated code 正好走了该安全 fallback。它是“pass 匹配但融合 kernel 未承接”，不是 NPU 编译不可用。

reduction 阻断来自接口不一致：

- PyTorch `torch/_inductor/lowering.py:7346` 的 `make_reduction()` 接受 keyword-only `strict_sum`；sum lowering 会传入该参数。
- 修复前 torch_npu `torch_npu/_inductor/lowering.py:144` 的覆盖函数没有 `strict_sum`，并在同文件以及 `torch_npu/_inductor/__init__.py` 替换上游函数。

T-011 已把 `strict_sum: bool = False` 作为 keyword-only 参数加入覆盖函数，并在 dump/non-dump 两个分支传入 `Reduction.create(strict_sum=...)`。没有修改 addmm pass、关闭 strict sum 或在测试中绕开梯度。

## 环境、设备与证据

- 环境：Conda `Pass`；PyTorch `2.14.0a0+git8e86e0a`；源码构建 torch_npu wheel `2.14.0a0+git83cc452`，以 `--no-deps` 安装；Triton Ascend 3.2.2；CANN 9.0.1。
- 所有命令从 `/home/z50063656/tmp` 启动，使用 `ASCEND_RT_VISIBLE_DEVICES=6` 和 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`。
- 物理 NPU 6 在开始前无运行进程。结束复查时 `npu-smi` 出现 PID 3579493、约 4.7 GB；该 PID 不属于报告中的 worker，且本进程命名空间的 `ps` 无法读取其详情。没有终止该进程，也没有在出现后继续执行 NPU 测试。本报告不使用本轮绝对耗时做性能结论。
- addmm inference：`results/p0_semantic_addmm_matrix_20260821/p0_gate_probe.json`。
- mm_plus_mm inference：`results/p0_semantic_mmplus_matrix_20260821/p0_gate_probe.json`。
- addmm full-bias backward：`results/p0_semantic_smoke_addmm_backward_full_bias_20260821/`。
- addmm vector-bias backward 阻断：`results/p0_semantic_smoke_addmm_backward_20260821/`。
- mm_plus_mm backward：`results/p0_semantic_smoke_mmplus_backward_20260821/`。
- 首次无设备权限的环境阻断被保留在 `results/p0_semantic_smoke_addmm_20260821/`，不得计为 pass 失败。
- T-011 最小 sum、原 blocker 和近邻回归：`results/t011_strict_sum_smoke_20260821/`、`results/t011_addmm_vector_backward_20260821/`、`results/t011_addmm_neighbor_regression_20260821/`、`results/t011_mmplus_backward_regression_20260821/`。这些运行在开始/结束均无其他进程的物理 NPU 6 上完成。

## 下一步

1. torch_npu `make_reduction(strict_sum=...)` 已完成源码 wheel 重建、`--no-deps` 安装和动态回归；addmm final verdict 为 `supported-beneficial`。
2. 对 mm_plus_mm different-K 先测当前安全 fallback 的 paired baseline，再决定是否值得实现支持独立 K1/K2 的 NPU Triton kernel。没有性能收益证据前不改 pass gate。
3. pad family 仍为 `unsupported`，下一阶段先测 NPU padding/slice + vendor GEMM 的收益，再设计 capability gate 或替代实现。
