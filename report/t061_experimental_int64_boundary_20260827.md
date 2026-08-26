# T-061 experimental int64 boundary downcast/dedup 审计（2026-08-27）

## 当前状态

状态：`correctness-failed-audit-fallback-proven-p020-design-pending`。installed wheel 的 int64
boundary downcast 在 int32 可证明值域内和 in-place case 可用，但超出 int32 或中间结果溢出时会
静默产生错误结果。audit-only `aten.mul` fallback 已恢复 exact correctness；尚未修改产品源码、
构建或安装 wheel，不能把审计 fallback 写成产品已修复。

目标功能族为 `TE-CG-010`：experimental codegen 把运行时 int64/fp64 tensor 参数对应的 Triton
签名降为 i32/fp32，launcher 在 kernel 边界创建临时 tensor；对于 input-only 参数可通过 weakref、
tensor `_version` 和目标 dtype 复用临时 downcast，对于 output/in-out 参数则在 launch 后写回原 dtype。

## 为什么必须先测正确性边界

这不是普通的“少一个 cast”优化。当前源码明确说明 boundary wrapper 只修复 ABI dtype mismatch，
不会修复 kernel 内部的 int32 溢出。因而至少有三层合同：

1. ABI 可用性：int64 输入/输出/in-place 图能否编译并执行，generated metadata 是否真实包含
   `downcast_args` 与 `*i32` 签名；
2. 值域正确性：int32 范围内必须和 NPU eager/CPU exact equal；`INT32_MIN/MAX` 邻界外必须显式
   证明是正确、fallback 或失败，不能因小值误差为 0 外推；
3. memo 正确性：相同未修改输入可复用；正常 in-place mutation 的 `_version` 变化必须使缓存失效，
   output/in-out 不得误复用旧 buffer，tensor 释放后 weakref 项必须清理。

## 已完成的静态链路

- installed wheel 与 current source 都在 `codegen/triton.py` 无条件把 `*i64` metadata 改为 `*i32`
  并登记 `downcast_args`；compute type 中的 `tl.int64` 也被全局改成 `tl.int32`；
- `npu_triton_heuristics.py:_make_launchers()` 看到 `downcast_args` 就包装 launcher；
- `dedup_downcast=True` 的 import-time snapshot 被 `_memoized_downcast()` 实际读取；
- `int64_boundary_cast=True` 虽被复制为 `npu_int64_boundary_cast`，但该名字在当前文件没有第二处读取，
  wrapper 安装路径仍是无条件的。因此它当前不是有效 ON/OFF gate，不能用 `config.patch()` 做正式
  boundary-cast A/B；generated marker 再次记录 source mention 只有 1，确认这是配置控制缺口；
- T-061 前的 source UT 只覆盖 in/out 在 downcast 前保留原内容、launch 后写回，没有覆盖超 int32 值域、
  input-only memo 命中/失效和真实 NPU generated code。

installed 三文件 SHA256：

- `npu_triton_heuristics.py`：`9dbcfa13...18fc`；
- `codegen/triton.py`：`fdf903db...2813`；
- `config.py`：`68ae6164...763b`。

runner：`inductor_pass_npu_audit/t061_experimental_int64_boundary_probe.py`，当前 SHA256
`1e9ea799...f113a`。

## 首轮预登记用例

所有 worker 从 `/home/z50063656/tmp` 发起，固定物理 NPU 1，source
`/home/z50063656/Benchmark/env.sh`，每个 compile 显式
`options={"npu_backend":"triton_experimental"}` 并校验 loaded backend。fresh compile 继续标注
T-022 audit-only launcher shim 边界。

1. `pointwise_input_output`：int64 `[4096]` 输入做 `x + 1`，覆盖 0、正负小值和
   `INT32_MIN/MAX` 内点；要求 dtype/shape/值 exact equal，且 generated code 证明 boundary cast；
2. `overflow_boundary`：分别放入 `INT32_MAX`、`INT32_MAX+1`、`INT32_MIN`、`INT32_MIN-1`，比较
   eager/compiled/CPU；任何 wraparound 都记 correctness failure，不放宽容差；
3. `inplace`：int64 input 做可观测 in-place update，验证输入内容先被带入临时 buffer、结果写回，
   且别名/版本语义与 eager 一致；
4. `memo`：同一 input-only tensor 连续调用、正常 `add_` 后重放、不同 tensor 和释放后的 weakref
   清理；先做 source UT，再决定是否需要设备 task/性能探针；
5. 只有功能合同通过后才测 dedup ON/OFF。性能采用 fresh paired 三轮、warmup 10/runs 100，记录
   host 与 NPU Event mean±stdev/P50/P99、task、首次编译和峰值内存。若边界正确性失败，先形成
   fallback/guard 设计，不用性能收益掩盖错误。

## 首轮 NPU 结果

三个原生 installed case 都确认 loaded backend 为 `triton_experimental`，generated wrapper 含
`downcast_args`、`*i64` 和 `*i32`，各生成 1 个 Triton kernel：

- int32 范围内 `x+1`：4096/4096 对 NPU eager 和 CPU exact equal；首次编译+运行
  `18.36 s`；
- int32 范围内 in-place `add_(1)`：返回值、被修改输入都 4096/4096 exact equal，返回值继续
  alias 原输入；首次编译+运行 `18.47 s`；
- 首版越界 `x+1`：2048/4096 mismatch，差值为 `-4294967296`。负边界有两项因模运算后碰巧相等，
  该中性漏检被保留，随后把复核表达式改为 `x*2`；
- 复核 `x*2`：`INT32_MAX`、`INT32_MAX+1`、`INT32_MIN`、`INT32_MIN-1` 四类输入全部错误，
  4096/4096 mismatch，差值为 `±4294967296`；首次编译+运行 `18.53 s`。

因此不能把“小值误差为 0”解释成 int64 语义完整支持。实际 kernel 的 load、常量和乘法都是
`tl.int32`，launcher 最后只把已环绕结果写回 int64。

## audit-only fallback 与 memo UT

fresh worker 仅从 `GENERATE_LIST` 移除 `aten.mul`，让现有 experimental fallback 注册接管同一
`x*2` 图。结果为 0 Triton kernel、generated wrapper 直接调用 `aten.mul`，4096/4096 对 NPU
eager/CPU exact equal；首次编译+运行 `2.78 s`。这只是方向验证，不是 source 实现，也不是正式
稳态性能结论，但已经证明无需且不应手写 Triton int64 kernel。

current source target suite 现为 6/6 通过：新增 memo UT 证明同一未修改 input-only tensor 复用同一
downcast；正常 `add_` 增加 `_version` 后生成新临时 tensor；不同 tensor 不串用；源 tensor释放后
weakref 项清理。该 UT 从 `/home/z50063656/tmp` 用 source-overlay 发起，实际加载 current source
codegen。当前测试文件 SHA256 为 `bde5da4a...657f7`；P-019 完成时的五测试 checkpoint hash
`76450d98...a58ed` 仍作为历史证据保留。

## P-020 设计边界

候选方向是“只对 int64 数据计算做 dtype-aware ATen fallback，保留普通浮点/低精度 Triton 和可证明
安全的索引用途”，不能简单从 `GENERATE_LIST` 永久删除整个算子包，也不能在 launcher 发现越界后
临时改跑另一个 kernel——launcher 不持有原 FX/ATen 语义。正式实施前至少要列出 add/sub/mul/
比较/reduction 的 dtype 路由，并验证 int64 embedding/index tensor 不被误伤。

## 当前停止边界

T-061 已形成 correctness blocker 与可行 fallback 方向，但还没有足够算子覆盖来安全修改产品源码。
下一步先做 dtype-aware fallback route prototype，覆盖 add/mul、比较、sum 与 int64 index negative；
功能矩阵通过后才登记具体 source diff 和 paired 性能。当前不手写 Triton int64 kernel——设备路径
本身就声明 AI Vector Core 不原生支持 int64 算术。
