# P0：NPU Gate Pass 用例设计

## 目标

P0 不直接证明“优化后更快”，先回答三个更基础的问题：

1. 上游 pattern 是否满足触发条件；
2. NPU 当前是否被明确 gate；
3. gate 后面缺少的是 lowering、codegen、精度还是性能收益。

当前 P0 有 7 条矩阵记录，其中两个 `_disable_*` 是控制点，实际测试 family 为 5 个。

## 公共执行要求

- 从 `/home/z50063656/tmp` 启动 fresh process。
- 每个 backend 独立进程：`default`、`ascendc`、`mlir`、`dvm`、`triton_experimental`；只测试实际安装且健康的 backend。
- 先记录当前默认 gate，不在首轮绕过 gate。
- 输出 eager 对齐、observer/counter、变换前后 FX 图、generated code、graph break/fallback、kernel 数。
- 功能通过后才测性能；性能统一 warmup 10、runs 100，并记录 CANN、SoC、mean/stdev/p50/p99、首次编译和峰值内存。

## P0-MMPLUSMM-001

最小语义：

```python
def fn(a, b, c, d):
    return a @ b + c @ d
```

触发条件来自 `is_valid_mm_plus_mm`：

- 开启 `max_autotune` 或 `max_autotune_gemm`；
- `a @ b` 与 `c @ d` 输出 M/N 相同；
- 两组 K 维分别合法；
- 四个输入都有 FakeTensor metadata。

参数组：

| 组 | dtype | shape | 用途 |
|---|---|---|---|
| A | fp16 | `(256,256) @ (256,256)` 两组 | 基础静态触发 |
| B | bf16 | `(257,255) @ (255,259)` 两组 | 非对齐 shape |
| C | fp32 | `(64,128) @ (128,96)` 两组 | dtype 覆盖 |
| D | fp16 | 输出 shape 不同 | extra-check 负例 |

首轮实测：default backend 在 `patch_pattern_mm_plus_mm()` 中被 gate，正例没有进入目标 fusion；`triton_experimental` 恢复 baseline 后正例已进入 `tuned_mm_plus_mm` 并生成 `extern_kernels._mm_plus_mm`。后续 candidate 对比必须按 backend 分开，不能再用“NPU 整体不支持”概括。

性能 baseline 是两个独立 mm 加 add；candidate 需分别比较 ATen/vendor、CATLASS/AscendC 和 Triton。矩阵类计算不默认采用手写 Triton。

## P0-PAD-MM-001 / P0-PAD-BMM-001 / P0-PAD-ADDMM-001

最小语义：

```python
def mm_fn(a, b):
    return a @ b

def bmm_fn(a, b):
    return torch.bmm(a, b)

def addmm_fn(bias, a, b):
    return torch.addmm(bias, a, b, beta=1.0, alpha=1.0)
```

公共触发条件：`shape_padding=True`、至少一个静态维、非零维、dtype/device 合法、至少一个 M/N/K 未按 backend alignment 对齐、Triton 可用。正常模式还会现场 benchmark；结构验证可用 `force_shape_pad=True` 强制经过 replacement，但不能用强制结果宣称性能收益。

参数组：

| family | 建议 shape | 负例 |
|---|---|---|
| mm | `(257,255) @ (255,259)` | 完全对齐 shape、零维、K=1 |
| bmm | `(4,257,255) @ (4,255,259)` | batch/contract 维不匹配 |
| addmm | bias `(259,)` + mm 上述 shape | bias 不可广播、dtype 不一致 |

首轮实测：`triton_experimental` 的 `_disable_pad_mm_pass()` 确实把 `shape_padding` 设为 False；default 虽保持 True，但上游 `check_device()` 仅接受 CUDA/XPU。补充诊断在 pattern 执行期间设置 `force_shape_pad=True` 后仍不能越过该 device gate，因此三个 family 当前对 NPU 为 `unsupported`。后续若设计 NPU capability gate，必须检查 padding、slice、额外内存和动态 shape，不允许只看 padded GEMM 本体时间。

## P0-ADDMM-001

最小语义：

```python
def fn(a, b, bias):
    return a @ b + bias
```

触发条件来自 `is_valid_addmm_fusion`：bias 可广播到 `(M,N)`，bias/a/b dtype 一致，图没有 flex-GEMM preserve 标记，并且不满足“应保持 unfused”的条件。

参数组：

| 组 | bias | alpha/beta | 用途 |
|---|---|---|---|
| A | `(N,)` | 1/1 | 常见 bias fusion |
| B | `(M,N)` | 1/1 | 完整 bias |
| C | `(1,N)` | 1/1 | broadcast |
| D | 标量 | 不适用 | 合法 eager 图，但不满足 tensor-bias addmm pattern |
| E | fp32 bias + fp16 mm | 1/1 | dtype 负例 |

NPU 已有 `kernel/mm.py:_register_npu_inductor_addmm()`，可选择 ATen、CATLASS 或 fallback。因此首要问题是 fusion 后是否进入有效 addmm choice，而不是重新实现 addmm。`triton_experimental` 默认通过 `_disable_addmm_fusion_pass()` 将已注册 handler 的 extra-check 置 False。

性能对比必须包含：原始 mm+add、fused addmm、CATLASS/vendor choice；同时统计 kernel 数、bias materialization 和 fallback。只有 pointwise/bias 承接成为瓶颈且已有实现无收益时，才评估 Triton。

## P0 覆盖扩展参数

`run_p0_gate_probe.py` 保持旧默认行为不变，并为 `addmm_fusion`、`mm_plus_mm` 增加以下显式 sweep：

| 维度 | 参数 | 当前代表值 |
|---|---|---|
| dtype | `--dtypes` | `float16,bfloat16,float32` |
| shape | `--shape-profiles` | `small=(32,64,48)`、`shape_a=(192,256,320)`、`unaligned=(191,255,319)`、`large=(512,768,640)` |
| layout | `--layouts` | `contiguous,transposed`；transposed 由反向 storage shape 转置得到真实非连续 stride |
| 动态图 | `--dynamic-modes` | `static,dynamic`；dynamic 还用 M/K/N 各增加 8 的第二组输入 replay |
| 单 pass 对照 | `--target-pass-modes` | `current,disabled` |
| 重复 | `--rounds` | 偶数轮反转 current/disabled 顺序 |

不直接执行所有参数的笛卡尔积。第一轮按四个 cohort 运行：dtype sweep 固定 shape-A/contiguous/static；shape sweep 固定 fp16/contiguous/static；layout sweep 只增加 fp16/shape-A/transposed；dynamic sweep 只增加 fp16/shape-A/contiguous/dynamic。每个 cohort 先 current 单轮验证正确性与目标图，再执行 current/disabled 三轮性能。

bf16 使用 `rtol=atol=3e-2`，fp16 使用 `1e-2`，fp32 使用 `1e-4`；容差会逐输出写入 JSON。非 P0 sweep family 若收到非默认 shape/layout/dynamic 或 disabled 请求，探针直接拒绝，避免生成含义错误的结果。

## 覆盖扩展与语义层结果

截至 2026-08-21，非笛卡尔代表功能矩阵 16/16 正确且目标图确认；paired performance 主矩阵 96/96 正确，mm_plus_mm dynamic 高样本复核 6/6 正确。addmm/default 的 8 个代表配置全部超过 10% p50 收益；mm_plus_mm/experimental 有 6 个超过门槛，transposed 和 dynamic 分别为 6.4% 与 8.74%，记为功能可用但性能 neutral。详见 `report/p0_sweep_function_matrix_20260820.md` 和 `report/p0_sweep_performance_20260820.md`。

代表网格证明了 dtype、shape、layout、dynamic 的基本能力，但不能替代语义测试。2026-08-21 已继续补齐以下正交用例，不重复性能笛卡尔积：

- addmm：`(M,N)` 与 `(1,N)` bias 都正确融合；fp32 bias + fp16 mm 正确保持未融合；
- mm_plus_mm：不同 K 会匹配 post-grad pattern，但当前 lowering 安全退回两个 mm 加 add；M/N 两个 broadcast 负例均正确不触发；
- mm_plus_mm same-K backward 的输出和 4 个输入梯度全部正确；addmm full-bias backward 的输出和 3 个输入梯度全部正确；
- addmm vector-bias backward 最初被 torch_npu `make_reduction()` 缺少 `strict_sum` 参数阻断。T-011 补齐并透传该参数、重建并安装源码 wheel 后，该用例的输出与三个输入梯度均正确；这证明它是已关闭的通用 reduction lowering 接口问题，不是 addmm pass 数值失败。

详细证据见 `report/p0_semantic_matrix_20260821.md`。reduction 接口兼容和 vector/row-bias backward 复测已经完成。T-012 的 mm_plus_mm different-K 当前/禁用 paired baseline 在 shape-A 与 unaligned 的 p50 变化为 -0.30%/+2.53%，结论为 neutral；T-013 profile 又确认两个 aclnnMm、一个 Triton add 和约 50–55 μs 步内 gap，理论上限为 17.68%/16.02%。现在只允许不接入源码的微原型，正式实现仍需真实 candidate 超过 10%。

## Gate 解除前置条件

任何临时绕过 gate 的实验都必须先在 `change_control.md` 增加 candidate 记录，写明：

- 具体 backend 和 capability 条件；
- 预计进入的 lowering/kernel；
- 失败时保持原 gate 的回退方式；
- 正确性范围和 paired benchmark 方案。

环境已于 2026-08-20 确认稳定，首轮 `default,triton_experimental` 共 20 个组合已经执行；结果见 `report/p0_gate_first_run_20260820.md`。本设计继续作为后续强制结构诊断和 paired benchmark 的验收合同。
