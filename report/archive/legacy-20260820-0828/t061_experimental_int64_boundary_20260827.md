# T-061 experimental int64 boundary downcast/dedup 审计（2026-08-27）

## 当前状态

状态：`installed-wheel-verified-correctness-restored-performance-characterized`。修复前 installed
wheel 的 int64 boundary downcast 在 int32 可证明值域内可用，但超出 int32 或中间结果溢出时会
静默产生错误结果。P-020 已在 experimental lowering 对显式数据 overload 增加 dtype-aware ATen
NPU fallback，并通过隔离 source、完整 wheel、专用 venv、目标 UT、安装态 NPU 8/8 功能矩阵和
三轮正确基线性能验证；未列入集合的其他 int64 op、其他设备与训练/反向不外推。

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

## 初始停止边界（历史 checkpoint）

文档提交 `a30979b` 时，T-061 只形成 correctness blocker 与可行 fallback 方向，尚没有足够算子
覆盖来安全修改产品源码；因此当时的下一步是 dtype-aware route prototype。后续各门禁和最终
installed-wheel 关闭结果按时间顺序追加在下文。这个 checkpoint 仍保留，用来区分“审计 fallback
证明可行”和“产品 wheel 已修复”；当前最终状态以文首及末节为准。

## P-020 dtype-aware route prototype 预登记

文档提交 `a30979b` 推送后继续执行，但仍不修改产品 lowering。fresh worker 将只在 backend
activation 时用审计 wrapper 包住目标 op 的现有 lowering：若 IR tensor input dtype 为 int64，调用
同一 ATen overload 的 `fallback_handler`；否则原样调用 upstream/experimental lowering。

首批矩阵：

| case | 风险 | dtype-aware route 预期 |
|---|---|---|
| int64 add/sub | 输入或结果越过 int32 | 0 Triton、ATen fallback、exact |
| int64 mul | 四边界与中间溢出 | 0 Triton、ATen fallback、exact |
| int64 gt | downcast 后符号翻转 | 0 Triton、ATen fallback、bool exact |
| int64 sum | 输入合法但 accumulator 溢出 | 0 Triton、ATen fallback、exact |
| FP32 mul control | 不能误伤常规数据计算 | 保持 1 Triton、无 fallback、exact |
| embedding int64 index control | index 角色不能被数据 dtype gate 误伤 | 保持原 lowering/codegen、exact |

每个 int64 case 先保存 native 结果，再跑 dtype-aware route；native correctness failure 是预期诊断，
但不能省略。只有六项 route 全通过，才考虑 P-020 source 实施；in-place/copy、dynamic 和更多 overload
在下一层覆盖。

## P-020 route prototype 结果与 source 门禁

首批和扩展矩阵均从 `/home/z50063656/tmp` 发起，固定物理 NPU 1，显式选择
`triton_experimental`。由于当前 PyTorch Python 来自源码树而完整 headers 来自 wheel，Triton
原生对照继续使用既有 T-022 launcher shim、wheel `torch/include` CPATH 和
`TRITON_DISABLE_PRECOMPILE=1`；ATen-only route 不依赖该 shim，但仍统一保留环境边界。

native correctness blocker：

| case | mismatch | generated structure |
|---|---:|---|
| add / sub | 各 2048/4096 | 各 1 Triton，i64→i32 boundary |
| mul | 4096/4096 | 1 Triton，i64→i32 boundary |
| gt | 2048/4096 | 1 Triton，降窄后符号翻转 |
| sum | 1/1 | 2 Triton，int32 accumulator 溢出 |
| copy | 2048/4096 | 1 Triton，纯搬运也被降窄 |
| in-place add | 返回值和输入各 2048/4096 | 1 Triton，写回 alias 但值错误 |

route prototype 结果：

- add/sub/mul、`gt.Scalar`、`gt.Tensor`、`sum.default`、`sum.dim_IntList` 均命中对应 ATen
  fallback，CPU/NPU eager exact；显式 `add.Scalar` 与双 tensor `add.Tensor` 也通过；
- in-place add 同时路由 `add.Tensor` 与 `copy_.default`，返回值、修改后输入和 alias 全部 exact；
- copy 路由 `copy.default` 后 exact。图中保留的 1 个 Triton 是 `zeros_like` 初始化 companion，
  不是 copy 本体，不能用“整图 0 Triton”替代逐 op route 检查；
- dynamic 双 tensor add 在同一 compiled function 上 `4096→4098` replay exact；
- FP32 mul route trace 为 original，保持 1 Triton、0 fallback、exact；
- 当前 910B2 的 embedding 原生就是 ATen fallback；native 与 route 生成代码 SHA256 同为
  `217991d4cd764d93a7cea699738b3b743ccf8c4f290bfc0359bca57bb560c3ed`，route trace 为空，
  证明 int64 index 未被 dtype compute wrapper 误伤。

审计过程中保留了三个中性修正：`value > 0` 实际 overload 是 `aten.gt.Scalar`；最初 fallback
计数正则把 `aten = torch.ops.aten` 别名误计为调用；embedding 结构断言错误地假设所有设备都生成
Triton。这三项都只修 runner 和合同，没有修改产品源码或覆盖失败产物。

因此 P-020 可以进入 source 实施：在 experimental fallback 注册结束后，对明确列出的 compute/
copy overload 安装幂等 dtype wrapper；只有 IR tensor input dtype 为 int64 时调用同 overload
`fallback_handler`，FP/低精度继续使用原 lowering，embedding/gather 等 index op 不在目标集合。
下一门禁是 source UT 与 source-overlay NPU 回归；通过前不构建/安装 wheel、不做性能结论。

## P-020 source 实施与功能门禁

正式修改仅位于 experimental lowering：在既有 fallback 注册结束后，对下列 overload 安装幂等
dtype wrapper：

- `add/sub/mul` 的 Tensor、Scalar；
- `lt/gt/ge/le/eq/ne` 的 Tensor、Scalar；
- `sum.default`、`sum.dim_IntList`；
- `copy.default`、`copy_.default`、`copy_.Tensor`。

wrapper 递归检查 args/kwargs 中的 IR tensor；发现 int64 才调用同 overload 的
`fallback_handler(..., add_to_fallback_set=False)`，否则保留注册前 lowering。没有移除
`GENERATE_LIST` 项，没有修改全局 Inductor lowerings，没有加入 embedding/gather，也没有生成
Triton int64 kernel。

source-overlay UT 从 `/home/z50063656/tmp` 发起，实际加载 current source lowering、heuristics 和
codegen，8/8 通过。source-overlay NPU 矩阵如下：

| case | generated route | correctness |
|---|---|---|
| int64 mul / gt / sum | 0 Triton、各目标 op 1 ATen fallback | eager/CPU exact |
| int64 copy | copy 本体 ATen fallback；允许 zeros_like companion Triton | eager/CPU exact |
| int64 in-place add | add 与 copy_ fallback | 返回、输入写回、alias exact |
| dynamic int64 add.Tensor | 同一 compiled function `4096→4098` | 两个 shape exact |
| FP32 mul control | 1 Triton、0 fallback | exact，浮点路径未误伤 |
| embedding int64 index | 保持设备原生 ATen embedding fallback | exact，代码 hash 未变 |

embedding native/source-overlay generated code SHA256 均为
`217991d4cd764d93a7cea699738b3b743ccf8c4f290bfc0359bca57bb560c3ed`。source implementation
不依赖 audit route trace；每个 int64 结果目录都核对实际加载的 source lowering 路径、fallback 调用、
Triton/downcast marker 与 exact 输出。

## P-020 正确基线三轮性能

原生 Triton int64 路径存在静默错误，不能作为性能基线。本轮比较语义正确的 eager ATen 与语义正确
的 source-overlay compiled fallback。固定 Ascend910B2 物理 NPU 1，输入为 1048576 个重复四边界
int64，表达式 `x*2`；每个 route/round 都是 fresh process，顺序为 eager→source、source→eager、
eager→source，warmup 10/runs 100。host 每样本包含同步，NPU Event 包围整个 callable；所有首调和
稳态输出均与 CPU exact。

| round | route | first ms | host mean±stdev / P50 / P99 ms | event mean±stdev / P50 / P99 ms | steady peak ΔB |
|---:|---|---:|---|---|---:|
| 1 | eager | 117.53 | 0.2961±0.0770 / 0.2859 / 0.3505 | 0.1879±0.0076 / 0.1872 / 0.2161 | 8389120 |
| 1 | source | 2862.79 | 0.4094±0.0706 / 0.3994 / 0.4510 | 0.3035±0.0076 / 0.3026 / 0.3224 | 8389120 |
| 2 | source | 2905.42 | 0.4157±0.0644 / 0.4049 / 0.4825 | 0.3150±0.0075 / 0.3134 / 0.3345 | 8389120 |
| 2 | eager | 114.13 | 0.2914±0.0778 / 0.2817 / 0.3364 | 0.1940±0.0066 / 0.1927 / 0.2140 | 8389120 |
| 3 | eager | 120.50 | 0.4734±1.5090 / 0.2934 / 1.1264 | 0.2131±0.0526 / 0.2012 / 0.4679 | 8389120 |
| 3 | source | 2814.15 | 0.4587±0.1259 / 0.4351 / 1.2388 | 0.3408±0.0776 / 0.3265 / 0.4217 | 8389120 |

跨三轮分别取各指标中位数：host mean/P50/P99 为
`0.2961/0.2859/0.3505 → 0.4157/0.4049/0.4825 ms`，绝对增加
`0.1197/0.1190/0.1320 ms`，相对 `40.4%/41.6%/37.7%`；NPU Event 为
`0.1940/0.1927/0.2161 → 0.3150/0.3134/0.3345 ms`，绝对增加
`0.1210/0.1208/0.1184 ms`，相对 `62.3%/62.7%/54.8%`。这是完整 compiled wrapper 加
ATen fallback 的端到端安全路径成本，不能表述成同一个 ATen kernel 本体回退；对这个约 0.2 ms 的
小算子，相对值会放大，绝对增量更能反映包装成本。两侧稳态峰值增量相同。

source 三轮 generated wrapper 均为 1 个 `aten.mul.Tensor`、0 Triton、0 `downcast_args`，SHA256
固定为 `8ed16917240cdafb3fcf01bd7b6a9eea3652301edf1d174a1ea33d80354d54f0`。source 首次
compile+run 中位数 `2862.79 ms`；eager 首调中位数 `117.53 ms` 单独记录，二者不是同类编译阶段，
不计算加速/退化比。

## P-020 静态门禁与当前边界

- source-overlay UT 8/8、四个相关 Python 文件 `py_compile`、产品目标 diff `git diff --check`
  通过；P-020 新增行无 Flake8、tab、版权提示。整文件仍有 P-020 之前的历史 lint 提示，未扩大
  修改范围；
- source 阶段当时 SHA256：lowering `35b8b098...5290`，test `36a9ff74...1701`，性能 runner
  `dfac86fb...506`，功能 runner `75e9cded...65e3`；
- source 阶段结论曾为 `source-verified-performance-characterized-wheel-pending-shared-tree`；后续已通过
  隔离 worktree 完成 wheel 和安装态验证，最终状态见下一节。

## P-020 隔离 wheel 与安装态关闭

为避免共享 source tree 的其他改动进入产物，从基线
`83cc452480c3546fd5cccf853bfe3a360ce9dbfc` 创建 detached worktree，只应用 P-020 的
lowering 和目标 UT 两个产品文件。构建前目标 diff 为 2 文件、139 行新增；构建期为 editable
PyTorch 的 `tools`、`torchgen/packaged` 和 `TorchConfig.cmake` 路径增加的临时辅助已全部撤销，最终
tracked 产品 diff 仍只有这两个文件，`git diff --check` 通过。

完整 wheel 构建成功：

- wheel：`torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`；
- SHA256：`028f678f39d6c353a408dfd621942b6ebd68a36469544e90b11da99b1a5c2822`；
- wheel 内、隔离 source、专用 venv 安装后的 `lowering.py` SHA256 均为
  `35b8b09897da8483a8dfb29e14845f1e8060320c4ed5f399e93c330964205290`；
- 安装位置为 `/home/z50063656/Pass/venvs/p020-installed`，没有覆盖共享 Benchmark 环境；实际 import
  路径与 marker 均指向该 venv，设备为 Ascend910B2；
- 安装态目标 UT 6/6 通过。

安装态 NPU 矩阵 8/8 通过：

| case | generated structure | installed 结果 |
|---|---|---|
| int64 mul / gt / sum | 各 1 ATen fallback、0 Triton、0 downcast | eager/CPU exact |
| int64 copy | copy 为 ATen fallback；zeros_like companion 为 1 Triton | eager/CPU exact |
| int64 in-place add | 2 ATen fallback、0 Triton | 返回、输入写回、alias exact |
| dynamic int64 add.Tensor | 1 ATen fallback、0 Triton | `4096→4098` 同图 replay exact |
| FP32 mul control | 0 fallback、1 Triton | exact，浮点路径未误伤 |
| embedding int64 index | 保持设备既有 ATen embedding fallback | exact，代码 hash `217991d4...0c3ed` |

原用例 int64 mul、gt、sum 在无 CPATH/CC shim 下直接通过。copy 的 companion Triton 无 shim 时因
editable PyTorch 暴露不完整 `torch/include` 而缺 `ATen/ATen.h`，保留该失败日志后使用既有 T-022
launcher shim 通过；这不影响 copy 的 P-020 ATen fallback，也没有把 shim 放入产品 wheel。

安装态三轮 fresh-process paired 与 source-overlay 结论一致。固定物理 NPU 1、
`numel=1048576`、warmup 10/runs 100，顺序为 eager→installed、installed→eager、
eager→installed：

| round | route | first ms | host mean / P50 / P99 ms | event mean / P50 / P99 ms | steady peak ΔB |
|---:|---|---:|---|---|---:|
| 1 | eager | 125.98 | 0.3195 / 0.2974 / 0.4442 | 0.1920 / 0.1913 / 0.2189 | 8389120 |
| 1 | installed | 2435.38 | 0.4100 / 0.3977 / 0.4767 | 0.3048 / 0.3031 / 0.3298 | 8389120 |
| 2 | installed | 2385.18 | 0.4421 / 0.4328 / 0.5210 | 0.3367 / 0.3348 / 0.3601 | 8389120 |
| 2 | eager | 115.98 | 0.2861 / 0.2785 / 0.3219 | 0.1897 / 0.1885 / 0.2028 | 8389120 |
| 3 | eager | 118.04 | 0.3182 / 0.2846 / 1.4389 | 0.2092 / 0.1885 / 0.2565 | 8389120 |
| 3 | installed | 2376.72 | 0.4223 / 0.4073 / 0.5125 | 0.3052 / 0.3022 / 0.3422 | 8389120 |

三轮指标中位数中，host mean/P50/P99 为
`0.3182/0.2846/0.4442 → 0.4223/0.4073/0.5125 ms`，绝对增加
`0.1040/0.1227/0.0683 ms`；NPU Event 为
`0.1920/0.1885/0.2189 → 0.3052/0.3031/0.3422 ms`，绝对增加
`0.1132/0.1146/0.1233 ms`。两侧稳态峰值增量相同。installed 三轮 generated wrapper
SHA256 均为 `8ed16917...54f0`，结构固定为 1 个 ATen mul fallback、0 Triton、0 downcast。

P-020 在已登记 overload 与 910B2/CANN 9.0.1 范围内由 FAIL→PASS，最终状态为
`installed-wheel-verified-correctness-restored-performance-characterized`。未列入显式集合的其他
int64 数据 op、其他设备型号和训练/反向仍不得据此外推；对应诊断模式已记录为 `P-COM-008`。
