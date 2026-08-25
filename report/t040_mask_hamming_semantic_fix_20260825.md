# T-040：mask arithmetic 与 sign-Hamming 语义修复及 NPU 功能闭环

## 结论

`masked_add_compose_pass` 与 `bool_cast_mul_to_where_pass` 原先会在浮点路径改变 IEEE
语义：前者不能保持“选中值再加正零”的 signed-zero 行为，后者会把 `false * Inf/NaN`
错误替换成常量零。两条 pass 现只在 bool/无符号 8 位/有符号整数 dtype 上改写；浮点与
复数保持原图。`sign_diff_hamming_fuse_pass` 在本轮特殊浮点、整数 keepdim 与 multi-user
边界中没有发现反例，暂不增加 dtype guard。

源码构建 wheel、源码态/安装态 76/76 FX 测试和 9/9 隔离 NPU fullgraph worker 全部通过。
九个 NPU worker 的图门禁、dtype/shape/stride/alias 合同、NaN/signbit 分类均通过，累计
`mismatch_count=0`。因此三条 pass 已形成“功能可用、性能待测”结论；尚不能写成
`supported-beneficial`。

## 为什么普通“数值误差为 0”不够

浮点 `+0.0` 与 `-0.0` 做普通相等比较时相等，但 `signbit` 可观察；NaN 又会让普通
`torch.equal` 返回 false，即使两边 NaN 分布完全一致。本轮先用 CPU eager/FX 构造反例：

- `where(m,a,0) + where(~m,b,0)` 在选中 `-0.0` 时会再加另一支 `+0.0`，结果可变成
  `+0.0`；直接改成 `where(m,a,b)` 会保留 `-0.0`。
- bool cast 后乘浮点 `x` 时，false 位置的 `0 * Inf` 与 `0 * NaN` 是 NaN；改成
  `where(mask,x,0)` 会成为 `0.0`。

所以 NPU worker 不只比较数值，还把 paired NaN 当作同类、单独比较 signbit，并继续检查
dtype、shape、stride、requires-grad 与输入 storage alias。这里的零 mismatch 表示这组完整
合同均通过，而不只是最大绝对误差恰好为零。

## 源码修改

修改范围严格限于 torch_npu Python pass 与其 FX 测试：

- `ascend_graph_pass.py:1725` 新增 `_EXACT_ZERO_MASK_DTYPES`，允许
  `bool/uint8/int8/int16/int32/int64`。
- `masked_add_compose_pass` 在两支 value dtype 都属于 allowlist 时才执行 where `2→1`、
  add `1→0`。
- `bool_cast_mul_to_where_pass` 在乘法另一输入属于 allowlist、且 cast dtype 与之相同时才
  执行 cast/mul→where。
- `test_dynamic_shape_fx_passes.py:592-812` 新增 9 项结构测试，覆盖整数正例、view chain、
  非互补和 multi-user 负例，以及 signed-zero、Inf、NaN、整数 keepdim。

没有修改 PyTorch、Triton、C++ 或 vendor op，也没有新增手写 Triton kernel。这里的问题是
pass capability 过宽；换一个 kernel 不能修复错误的等价关系。

## Wheel 与回归

- 构建入口：`build_t023_torch_npu_wheel.py`，构建日志为
  `results/t040_torch_npu_wheel_build_20260825.log`。
- 新 wheel：`src/torch_npu/dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，
  SHA256 `b273aeedcb9d1367de65328bea78a448ff1eb81fa4a85dca3f910e556c7b2460`。
- T-038 回退 wheel：`artifacts/torch_npu_t038_before_t040_mask_ieee_fix.whl`，SHA256
  `dffad49056538fc4250b444b2c40a619db3b0897b00f8906f53757a857b167d8`。
- 新 wheel 以 `pip install --no-deps --force-reinstall` 安装；日志为
  `results/t040_torch_npu_wheel_install_20260825.log`。
- source-PYTHONPATH、关闭 backend autoload 的完整文件为 76/76，通过日志为
  `results/t040_source_fx_tests_20260825.log`。
- installed site-packages、正常 backend autoload 的完整文件为 76/76，通过日志为
  `results/t040_installed_fx_tests_20260825.log`。

第一次 installed 测试错误地同时关闭 autoload 又在测试 stub 后显式加载 native torch_npu，
导致 `_npu_dtype_cast` schema 重复注册，保留在
`results/t040_installed_fx_tests_autoload_disabled_failure_20260825.log`。正常 autoload 下全过，
因此它是测试启动方式错误，不是 pass 或 wheel 失败。

## NPU fullgraph 证据

所有 worker 从 `/home/z50063656/tmp` 启动，使用物理 NPU1、default backend、inference、
fresh output/cache。运行前后 NPU1 均无其他进程。由于当前 PyTorch/Triton Ascend launcher
header 合同仍未闭环，统一使用 T-022 已登记的 C++20/header audit shim；所以结论标注为
development/audit-shim，不能替代未来无 shim smoke。

| profile | 关键图门禁 | 结果 |
|---|---|---|
| masked integer positive | where `2→1`，add `1→0` | complete，mismatch 0 |
| masked non-complement | where `2→2`，add `1→1` | complete，负例未误改 |
| masked float signed-zero | where `2→2`，add `1→1` | complete，NaN/signbit 一致 |
| bool integer direct | cast `1→0`，mul `1→0`，where `0→1` | complete，mismatch 0 |
| bool integer view | cast `1→0`，unsqueeze `1→1`，mul `1→0`，where `0→1` | complete |
| bool float nonfinite | cast `1→1`，mul `1→1`，where `0→0` | complete，NaN/signbit 一致 |
| hamming float special | sign/relu/sub/abs `2/2/1/1→0`，gt/ne `0/0→2/1` | complete |
| hamming integer keepdim | 同上，sum `1→1` | complete，dtype/shape 一致 |
| hamming multi-user | sign/relu `2/2→2/2`，gt/ne `0/0→0/0` | complete，负例未误改 |

聚合证据位于
`results/t040_mask_hamming_compile_20260825/aggregate/aggregate.json`。第一次 masked integer
worker 的 `CPATH` 误指 editable 源码 include view，launcher 报缺少 `ATen/ATen.h`；失败目录
原样保留，改用 site-packages wheel headers 后在新目录通过，不计产品失败。

## 下一步

性能优先级暂定为：

1. `sign_diff_hamming_fuse_pass`：源码层把 sign/relu/sub/abs 链换成 gt/ne，算术复杂度下降
   最大，先做单 pass 三轮 paired 与 task profile。
2. `masked_add_compose_pass`：整数 safe path 删除一个 where 和 add；若 baseline 已被 scheduler
   融成单 kernel，收益可能只来自 kernel 内指令减少。
3. `bool_cast_mul_to_where_pass`：direct 与 view-chain 各做代表 shape；若两边始终 1 task 且
   p50 未超过 10%，登记 neutral，不再手写功能重复的 Triton。

性能测试必须只关闭目标 pass、三轮交错 fresh process、warmup 10/runs 100，并报告
mean±stdev、p50/p99、task 数与显存。首次 compile+run 不参与稳态 verdict。
