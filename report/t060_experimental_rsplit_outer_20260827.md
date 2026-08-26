# T-060 experimental `rsplit_outer` 审计（2026-08-27）

## 当前结论

状态：`source-verified-beneficial-wheel-pending-environment-monitor`。

当前 wheel 的 `rsplit_outer=True` 不是完全不可用：静态 scalar sum 能原生命中，生成
partial+combine 两个 Triton kernel 并正确执行。但目标 OUTER/RMSNorm dweight 图在
`Reduction.num_splits()` 阶段得到 `ReductionHint.DEFAULT`，而
`_npu_rsplit_outer_applicable()` 只接受 `INNER/OUTER`，所以配置虽然为 True，真实图仍保持一个
普通 reduction kernel。

仅在审计 worker 内把该图的 `DEFAULT` 强制解释为 `OUTER` 后，既有 rsplit codegen 无需重写
Triton 就能正确生成并执行；最终 P-019 真实 source-overlay 三轮 device P50/P99 中位改善
`29.93%/29.94%`。因此当前优先修复点是 reduction hint 与 experimental gate 的衔接，不是手写
替代 kernel。P-019 随后只在 rsplit 局部接受 DEFAULT，并保留其余 gate；target UT、source-overlay
static/dynamic NPU 和阈值负例已通过，尚未构建或安装 wheel。

## 机制与目标

`triton_experimental/codegen/triton.py:_npu_rsplit_outer_applicable()` 面向窄输出、长 sum
reduction，把原单 kernel 拆为：

1. partial kernel：按 r 轴跨核切分，每核把一份 `[x_total]` 偏和写入 workspace；
2. combine kernel：沿实际 core 数归约 workspace，写真实输出。

Ascend 910B2 本轮生成代码使用 `total_thread=48`，RMS case 的 workspace 为
`48 × 2048 = 98,304` 个 FP32 元素。P-019 同步把旧源码中固定写死的“40 cores”说明改成
设备无关的 `num_cores` 表述；设备核数仍以生成代码为准。
gate 还要求单一 sum、非 welford、唯一 reduction output、`r>=2048`、`r>=x` 和受限 x 大小。

## 环境与方法

- 环境：`/home/z50063656/Benchmark/env.sh`，CANN 9.0.1，Ascend 910B2，固定物理 NPU 1；
- 版本：torch `2.14.0a0+git8e86e0a`，torch_npu `2.14.0a0+git83cc452`；
- backend：每个 fresh worker 显式
  `options={"npu_backend":"triton_experimental"}`，结果验真
  `_InductorNpuRegistry._loaded_backend == "triton_experimental"`；未显式传参时环境默认仍是
  `default`；
- A/B：在 `codegen.triton` 首次导入前设置 `rsplit_outer=False/True`，规避模块级 config 快照；
- runner：`inductor_pass_npu_audit/t060_experimental_rsplit_outer_probe.py`，当前 SHA256
  `fc3c25c1...570f04`；
- fresh Triton launcher 仍需 T-022 audit-only wheel-header/C++20/CANN shim。以下结果可用于
  pass/codegen/device 归因，不能写成正式无 shim 环境通过；
- 性能：每个 worker warmup 10、runs 100，同时记录 host 与 NPU Event
  mean±stdev/P50/P99；顺序 `OFF1/ON1/ON2/OFF2/OFF3/ON3`。

## 原生 wheel 可达性

### 成功入口

静态 FP32 `[4096].sum()`：ON 原生命中 `ReductionHint.INNER`，生成 2 个 kernel，包含
`rsplit_nsplit/rsplit_id/ws_ptr`；对 NPU eager 最大绝对误差 `7.6294e-06`，对 CPU
`1.3351e-05`，均通过 `rtol=1e-3, atol=1e-2`。这证明 pass 与设备 codegen 本体可用。

### 中性未命中尝试

- RMSNorm dweight 同构图：`[4096,2048] -> [4096,16,128]`，乘两个输入后 `sum(0)`；
- outer sum：`[4096,1,256].sum((0,1))`，dynamic 与 static 均测试；
- inner sum：`[16,4096].sum(-1)` 与 `[256,4096].sum(-1)`。

上述 ON 都正确执行，但仍为 1 kernel、无 workspace。只读 gate trace 对非标量 inner case记录：

- `Reduction.num_splits()` 返回 `ReductionHint.DEFAULT, 1`；
- reduction node 与 feature hint 均为 `DEFAULT`；
- `_npu_rsplit_outer_applicable()` 因 hint gate 返回 False。

初始 RMS dynamic `4096→4097` 的 ON/OFF 正确性均通过，但没有触发 rsplit；它只能证明普通
reduction 支持 dynamic，不能证明 rsplit dynamic 可用。这些结果均保留，不算功能失败，也不从
scalar success 外推为 OUTER 可用。

## audit-only 候选修复验证

runner 的 `--force-hint outer` 只在当前 worker 内把 `SIMDKernelFeatures` 的 `DEFAULT` 改为
`OUTER`，不修改产品源码或 installed wheel。该 overlay 用于回答“既有 rsplit 实现能否承接目标
图”，不能作为交付方案。

- outer sum `x=16,r=4096`：2 kernels，结构 marker 完整；对 NPU eager/CPU 最大误差
  `1.7166e-05/2.2888e-05`；
- RMSNorm dweight `x=2048,r=4096`：2 kernels，partial 写 `98,304` 元素 workspace，combine
  以 `r0_numel=48` 归约；对 NPU eager/CPU 最大误差不超过 `6.1035e-05`；
- OFF 为单 kernel；ON 为 partial+combine，实际 loaded backend 始终是
  `triton_experimental`。

这说明不需要复制已有实现去手写新的 Triton kernel。候选产品修复应只在 experimental rsplit
applicability gate 内局部接受 DEFAULT，并继续依赖单 sum、唯一输出和 x/r 阈值等既有结构 gate；
不能全局强改所有 `SIMDKernelFeatures.get_reduction_hint()`。

## P-019 source 实施与覆盖

文档先行和 audit-only 最小验证完成后，目标 `triton.py` 确认为 clean，P-019 只修改两处：

- `torch_npu/_inductor/triton_experimental/codegen/triton.py`：允许
  `_npu_rsplit_outer_applicable()` 接受 `ReductionHint.DEFAULT`；没有修改全局 hint、partial/combine
  kernel、workspace、阈值或其他 reduction lowering；同处过时的固定 40 核注释改为设备无关表述；
- `test/_inductor/test_triton_experimental_native_bert_regressions.py`：新增 DEFAULT positive，并在同一
  test 验证 `r=1024`、非 sum、`x>48*256` 和 `OUTER_TINY` 仍拒绝；原有 fused pointwise output 与
  nested reduction 负例继续执行。

P-019 验证 checkpoint 的源码/测试 SHA256 为 `9faa1655...0d3a5`/`76450d98...a58ed`；随后只把同处
gate 说明改成准确的 `ncfg.rsplit_outer` 名称，并由 T-061 增加 memo UT，current SHA256 为
`93ad31c1...fdf73`/`bde5da4a...657f7`。从 `/home/z50063656/tmp` 发起的 P-019 source-overlay
目标 UT 为 5/5 通过，加入 T-061 后 current suite 为 6/6；lintrunner 在 Benchmark 环境不可用
（command not found），已完成 `py_compile` 与 `git diff --check`。

设备覆盖结果：

- current-source、无 audit hint 参数：RMS static 与 dynamic `4096→4097` 都命中 2 kernels；dynamic
  replay 对 NPU eager 最大误差 `9.1553e-05`；
- current-source threshold negative：outer sum `r=1024,x=16` 正确且保持 1 kernel；
- 等价的最小 gate overlay 预验证：FP16/BF16 RMS positive 为 2 kernels，最大误差分别
  `0.125/1.0`，在预登记 `rtol/atol=0.01/0.1` 与 `0.03/0.5` 合同内；
- negative：`r<x`、amax、variance/multi-reduction 均正确且保持 1 kernel；variance trace 为
  2 个 reduction nodes，因此原唯一 reduction gate 生效。

后两类是 source 修改前的等价 gate overlay 证据；正式 installed wheel 回归时仍应统一重跑，不能
用它们替代 wheel 结果。

## 三轮代表性能

固定 FP32 `rows=4096, groups=16, inner=128, x=2048`。OFF 为当前 wheel 原生单 kernel；ON 为
P-019 current-source overlay 的既有 partial+combine，没有 force-hint/allow-default audit 参数。
三轮 backend、source path、结构和正确性重复通过。早先全局强制 OUTER 的三轮中位改善为
`30.55%/31.66%`，但它同时改变 partial heuristic，现只保留为候选上限，不作为 source verdict。

### NPU Event

| round | OFF mean±stdev ms | ON mean±stdev ms | P50 OFF/ON ms | P99 OFF/ON ms | P50 改善 | P99 改善 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.34691±0.00969 | 0.24140±0.02197 | 0.34673/0.24294 | 0.38056/0.26662 | 29.93% | 29.94% |
| 2 | 0.35371±0.01027 | 0.25348±0.00889 | 0.35088/0.25144 | 0.38878/0.27968 | 28.34% | 28.06% |
| 3 | 0.35444±0.01196 | 0.24683±0.01329 | 0.35169/0.24377 | 0.40536/0.27568 | 30.69% | 31.99% |

三轮 device P50/P99 改善中位数为 `29.93%/29.94%`，且三轮同向。

### Host

| round | OFF mean±stdev ms | ON mean±stdev ms | P50 OFF/ON ms | P99 OFF/ON ms | P50 改善 | P99 改善 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.87937±0.82712 | 0.29607±0.01635 | 0.42260/0.29339 | 2.61278/0.33494 | 30.58% | 87.18% |
| 2 | 0.40439±0.01635 | 0.30432±0.01990 | 0.40007/0.30129 | 0.45471/0.33939 | 24.69% | 25.36% |
| 3 | 0.40656±0.01687 | 0.30517±0.02649 | 0.40251/0.29751 | 0.48639/0.41689 | 26.09% | 14.29% |

host P50/P99 改善中位数为 `26.09%/25.36%`。round 1 OFF 的
`mean±stdev=0.87937±0.82712 ms` 与 `P99=2.61278 ms` 是真实长尾，保留但不用于夸大收益；device
三轮结果仍稳定。

## 成本与边界

- max allocated：OFF `134,788,608 B`，ON `135,182,336 B`，增加 `393,728 B`；生成代码中的
  workspace 本体为 `393,216 B`，其余为分配/对齐差；
- 强制禁缓存的首次编译+运行中位数：OFF `23.31 s`，ON `43.26 s`，约增加 `85.6%`；
- 三轮 ON 对 NPU eager 最大误差 `6.1035e-05`，OFF 为 `5.3406e-05`；两者对 CPU 最大误差均
  不超过 `4.5776e-05`，均在预设 FP32 合同内；
- dynamic、FP16/BF16、r<2048、r<x、非 sum 和 variance/multi-reduction 已有设备或等价 overlay
  证据；超宽 x、`OUTER_TINY`、fused-output 和 nested-reduction 已由 source gate UT 覆盖，installed
  wheel 尚未统一复验；
- 本轮已做 P-019 两文件纯 Python source 修改，没有构建或安装 wheel，也没有手写 Triton 替身。

## 下一步

1. 在共享 source 其他 diff 可隔离后，按既定流程构建 torch_npu wheel并 `--no-deps` 安装；
2. installed wheel 统一重跑 static/dynamic、三 dtype、四类 negative 和三轮 paired；
3. 正式无 shim launcher 环境复验仍单独保留，P-019 不能顺带打包 P-014/P-016/P-017/P-018 或
   其他人的未安装改动。
