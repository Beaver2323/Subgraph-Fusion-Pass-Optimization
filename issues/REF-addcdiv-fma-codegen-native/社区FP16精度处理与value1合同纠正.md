# 社区 FP16 精度处理与 value=1 合同纠正

> 更新时间：2026-09-09 18:05:53 CST（UTC+08:00）
> PyTorch 基线：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> NPU 验证后端：`triton_experimental`

## 1. 结论

本项目的功能正确性主判据是：同一组 NPU 输入下，`torch.compile` 输出与 NPU eager 输出对齐。
CUDA 的实现用于借鉴“明确计算 dtype、运算顺序与舍入点”的修复方法，不作为要求 NPU 输出逐位
复制 CUDA 输出的替代 oracle。若 CUDA eager 与 NPU eager 的低精度舍入语义不同，应保留 NPU
后端专属 lowering，并用 NPU eager 关闭功能精度。

社区对 CUDA `addcdiv` 低精度的处理不是“除法后落 FP16、乘法后再落 FP16”，而是与 CUDA
eager 的专用 kernel 对齐：FP16/BF16 输入提升到 FP32，在 FP32 中完成除法和乘加，最后把输出
写回低精度。Inductor 用 `div_rn` 固定除法舍入，并对 `value!=1` 使用 `tl.fma` 固定乘加语义。

`value=1` 是独立分支：decomposition 消去乘一，只剩 `div -> add`；社区 bitwise case 要求结果
正确，但不要求也不会命中 `div -> mul -> add` FMA pattern。因此 GPU 与 NPU 的正式预期均为
`addcdiv_fma_fused=0`。

## 2. CUDA eager 如何计算 FP16/BF16

CUDA kernel 对非复数类型使用 `accscalar_t`。当 `scalar_t` 是 FP16/BF16 时，CUDA 上的
`acc_type` 是 FP32：

```cpp
// PyTorch: aten/src/ATen/native/cuda/PointwiseOpsKernel.cu:176-184
AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcdiv_cuda", [&]() {
  // If scalar_t is fp16 or bfloat16, cast scalar to float
  // and do math in fp32 for better accuracy.
  using accscalar_t = at::acc_type<scalar_t, true>;
  auto alpha = value.to<accscalar_t>();
  gpu_kernel(iter, [alpha] GPU_LAMBDA(
      scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
    return pointwise_op_impl<accscalar_t>(
        a, b, c, alpha, std::divides<accscalar_t>());
  });
});
```

辅助函数固定运算顺序：

```cpp
// PyTorch: aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh:9-28
if (alpha == opmath_t(1)) {
  return input + op(tensor1, tensor2);
}
if constexpr (std::is_floating_point_v<opmath_t>) {
  return std::fma(alpha, op(tensor1, tensor2), input);
}
```

所以 CUDA FP16 的语义可概括为：

```text
value=1:  fp16(fp32(self) + fp32(t1) / fp32(t2))
value!=1: fp16(fma(fp32(value), fp32(t1) / fp32(t2), fp32(self)))
```

中间 quotient 和 product 不会像 NPU `aclnnAddcdiv` 的实测行为那样分别落回 FP16。

精确到舍入点，`value!=1` 的 CUDA FP16 eager 是：

```text
a32     = exact_cast_fp16_to_fp32(self)
b32     = exact_cast_fp16_to_fp32(tensor1)
c32     = exact_cast_fp16_to_fp32(tensor2)
value32 = cast_to_fp32(value)
q32     = RN_FP32(b32 / c32)
r32     = RN_FP32_FMA(value32, q32, a32)
out16   = RN_FP16(r32)
```

FP16 有限值提升到 FP32 是精确转换。因此这条路径只有除法的 FP32 舍入、FMA 的一次
FP32 融合舍入和最终 store 的 FP16 舍入，没有 quotient/product 的 FP16 中间舍入。

## 3. Inductor 如何复现 CUDA eager

专用 lowering 对 CUDA/XPU 浮点使用 `div_rn`；只有 `value!=1` 使用 FMA：

```python
# PyTorch: torch/_inductor/lowering.py:8581-8608
use_fma = (
    dtype.is_floating_point
    and device is not None
    and device.type in ["cuda", "xpu"]
)

if use_fma:
    t1_div_t2 = ops.div_rn(t1_val, t2_val)

if value == 1:
    return ops.add(self_val, t1_div_t2)

if use_fma:
    return ops.fma(value_expr, t1_div_t2, self_val)
```

FP16/BF16 在 Triton 中使用 FP32 compute type，最终 store 再转换成张量 dtype。这里的
`div_rn + fma` 是对 CUDA eager FP32 运算链的约束，不是插入两个 FP16 中间舍入点。

`div_rn` 防止除法被改写成倒数近似或发生重结合；`tl.fma` 则避免普通
`mul -> add` 在两步之间多一次 FP32 舍入。所以“不显式做 FP16 中间舍入”仍能
与 eager 位级一致，原因不是 FMA 自动修正了任意误差，而是 eager 和 compiled 有意使用了
同一个 FP32 运算序列和相同舍入边界。

需要区分两层保证：社区源码通过 `accscalar_t=float`、`std::divides<float>` 和
`std::fma` 在设计上对齐；Tracker 又在 A100 上用只替换 dtype 的 FP16 派生用例断言
`torch.equal(compiled, eager)`、counter=1 且生成代码同时包含 `tl.fma`/`div_rn`。

## 4. 社区官方测试实际覆盖什么

社区 bitwise case 使用没有显式 dtype 的 `torch.randn`，所以当前只覆盖 FP32：

```python
# PyTorch: test/inductor/test_torchinductor.py:19023-19035
s = torch.randn(64, 64, device=GPU_TYPE)
t1 = torch.randn(64, 64, device=GPU_TYPE)
t2 = torch.randn(64, 64, device=GPU_TYPE).abs().clamp(min=0.1)

for value in (1.0, 2.0):
    ...
    self.assertEqual(fn(s, t1, t2), torch.addcdiv(...))
```

另一条 codegen case 固定 `value=2.0`，才要求 counter=1、`tl.fma` 和 `div_rn`：

```python
# PyTorch: test/inductor/test_torchinductor.py:19039-19064
return torch.addcdiv(s, t1, t2, value=2.0)
...
self.assertEqual(counters["inductor"].get("addcdiv_fma_fused", 0), 1)
self.assertIn("tl.fma", code)
self.assertIn("triton.language.div_rn", code)
```

Tracker 的 A100 FP16/BF16 测试是“只改变 dtype”的派生补测，并非社区原生 FP16 测例。该补测
证明冻结 commit 与给定输入下 GPU FP16 能按上述 CUDA 合同逐位通过，但不能改写官方覆盖范围。

## 5. `emulate_precision_casts` 与本问题的关系

社区另有通用开关 `torch._inductor.config.emulate_precision_casts`，用于多个低精度 pointwise
算子融合后，中间 downcast/upcast 被消除的场景：

```python
# PyTorch: torch/_inductor/config.py:3159-3172
# eager computes bf16/fp16 by upcasting inputs to fp32 and downcasting after.
# When two low precision operators are fused, Inductor elides the intermediate
# downcast-upcast. This knob preserves them to emulate eager numerics.
emulate_precision_casts = ...
```

启用时，原用户程序里的低精度 pointwise 节点会带
`low_precision_pointwise_barrier`，lowering 在节点输出处插入低精度 downcast/upcast，并关闭
Triton 浮点融合。这个机制最初用于调试/复现 eager 低精度数值，默认关闭。

它不能被描述为社区 `addcdiv` 官方 FP16 修复：`addcdiv` 是 decomposition，社区特意不把分解
内部节点当作用户级 pointwise barrier；而且 addcdiv 的官方测试没有启用这个开关。专用 CUDA
addcdiv lowering 仍以 CUDA eager 的 FP32 accumulate + 最终低精度 store 为准。

## 6. 对 NPU 修复的约束

NPU eager 的实测舍入边界不同，因此 `value!=1` 必须保留后端专属 lowering：

```text
q = fp16(t1 / t2)
p = fp16(q * fp16(value))
out = fp16(self + p)
```

但这只是 NPU 低精度 lowering 差异，不能改变 FMA pattern 的结构边界：

```text
value!=1: div -> mul -> add，可命中，counter=1
value=1:  div -> add，预期不命中，counter=0
```

2026-09-09 已在实际 Ascend 910B2 上确认 NPU FP16 `value=1` 的普通 `div -> add` 路径
修复前与 eager 不一致：counter=0，但 1176/4096 个元素不同，最大绝对误差
`0.015625`。该问题按独立低精度 pointwise/lowering 回归处理，没有伪造 FMA
pattern 命中。修复后 quotient 在 div 与 add 之间显式落回 FP16，真机 mismatch=0、
最大误差=0、counter 仍为 0。

## 7. 本轮纠正

- 删除 torch_npu 候选源码及 tracker candidate 中的 `add(div)` 补充重融合；
- runner 将 `value=1` 的 expected fusion 改为 0；
- FP16 FMA 修复验证只统计 `value=0.3/2/7.7` 的 `value!=1` 路径；
- 旧 value=1 counter=1 的 FX/IR/codegen 和 JSON 保留，但明确标记为不计数历史；
- 机器可读纠正入口：
  `results/current/AU-post-grad-fuse-addcdiv-to-fma/value_one_contract_correction_20260908.json`。

19:49 CST 的 `aclInit 507008` 沙箱尝试仍保留为不计数环境失败。随后在 NPU 可见执行层完成
真机重跑：三个 `value!=1` 与两个 guard 全部通过；另起 fresh process 的 FP16 `value=1`
执行到生成 kernel 并暴露上述 correctness regression。随后的修复及六进程再验证也是
代码在实际设备上的执行结果，不是设备不可见造成的假通过。

## 8. 进度矩阵中的社区对齐结论

本单元登记为 `PARTIAL_ALIGNED`，不是笼统的“已与 CUDA 完全一致”：

| 范围 | 结论 |
| --- | --- |
| 已对齐 | FP32 `value=1` 在 GPU/NPU 均不命中 FMA pattern且正确；FP32/BF16 的 value!=1 命中与正确性合同成立 |
| 实现差异 | GPU FP16 为 `div_rn+FMA`；NPU FP16 为匹配 NPU eager 显式保留两个低精度舍入边界 |
| 未决范围 | FP16 `value=1` 的 GPU dtype/value 邻接 reference |
| 处置 | 保留后端专属 FP16 value!=1 lowering；value=1 用普通 div lowering 恢复 quotient 舍入并保持 counter=0 |

机器可读原件位于
`results/current/AU-post-grad-fuse-addcdiv-to-fma/comparison_result.json`，当前进度展示位于
`report/current_acceptance_unit_matrix.md`；完整范围字段保存在同目录 CSV。
