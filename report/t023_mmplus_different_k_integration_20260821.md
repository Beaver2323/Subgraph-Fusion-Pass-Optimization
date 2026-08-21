# T-023 mm_plus_mm different-K 正式接入报告（2026-08-21）

## 结论

T-023 已把 different-K `mm + mm` 的单 Triton task 候选接入 torch_npu default Inductor backend，并随当前源码构建 wheel 使用 `--no-deps` 安装。首批静态 shape 功能与性能验证通过，但严格显存门槛失败，因此结论是：

`supported-beneficial-opt-in-with-memory-tradeoff`。

实现继续默认关闭，不替换 large、dynamic、空维、same-K 或任意 stride 路径，并始终保留原有 extern fallback。当前混合 editable PyTorch/Triton Ascend/header 合同还需要 audit-only launcher shim 才能 fresh compile candidate；所以正式交付仍有一项“匹配 headers 的无 shim 环境复验”，不能把本报告解释为当前共享环境已完全产品化。

## 实现范围

产品改动位于 `/home/z50063656/Pass/src/torch_npu`：

- `torch_npu/_inductor/config.py`：增加默认关闭的 `TORCHINDUCTOR_NPU_ENABLE_DIFFERENT_K_MM_PLUS_MM`，默认 output 上限为 131072 elements。
- `torch_npu/_inductor/fx_passes/post_grad.py`：注册 NPU-only duplicate pattern；限制 default backend、静态正二维 shape、K1!=K2、同 device/dtype、row/column-major stride 和 output 上限。
- `torch_npu/_inductor/kernel/mm_plus_mm.py`：新增独立 K1/K2 loop 的 `NPUTritonTemplate`；128×128×128 tile；extern choice 始终排在 fallback 列表中。
- `torch_npu/_inductor/kernel/__init__.py` 与 `torch_npu/_inductor/__init__.py`：安装 kernel 和幂等 pattern 注册。
- `test/_inductor/test_mm_plus_mm.py`：结构、gate、fallback 和幂等测试，共 6 项。

没有修改 PyTorch、Triton Ascend 或 C++ dispatcher。T-011 的 `strict_sum` 修改被保留在同一个 wheel 中。

修复版 wheel：

- 文件：`dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`
- SHA256：`d0ee10794f8cb63d528c86f27294a2a52a4b8b5f484eb6be53323d22b2157718`
- 安装：`pip install --no-deps --force-reinstall`
- installed-package 结构测试：6/6 通过

## 功能验证

| 范围 | 结果 | 证据含义 |
|---|---|---|
| rollout off 正例 | 正确，无 template event | 默认行为不变 |
| rollout on row-major | 正确，实际执行单 Triton task | 首批主路径可用 |
| rollout on column-major | 正确，实际执行单 Triton task | 已登记 stride 可用 |
| AOTAutograd backward | output 与 A/B/C/D 四个梯度误差均为 0 | forward pattern 可由独立 backward 图承接 |
| same-K | 不追加 template，正确 fallback | 不改变上游 same-K 行为 |
| large | 不追加 template | output 上限有效 |
| empty | 不追加 template | 非空 gate 有效 |
| arbitrary stride | 不追加 template | 不冒充通用 layout 支持 |
| dynamic first/replay | 不追加 template，结果正确 | 首批只支持 static |

template 选择通过进程局部审计 wrapper 记录；deferred `MultiTemplateBuffer` 场景按最终 task 验证，避免把 selector 返回 `None` 误判为 fallback。成功 case 按图模式规则不读取 `output_code.py`；只有失败诊断读取本轮生成代码。

## 集成性能

环境为 Python 3.11.15、PyTorch `2.14.0a0+git8e86e0a`、torch_npu `2.14.0a0+git83cc452`、Triton module 3.2.0、CANN 9.0.1、Ascend910B2。每个 baseline/candidate/round 都是独立 fresh process；warmup 10、runs 100、三轮交错；所有正确性 max/mean absolute error为 0。

| cohort | baseline p50 中位数 | candidate p50 中位数 | p50 改善 | baseline p99 中位数 | candidate p99 中位数 | p99 改善 |
|---|---:|---:|---:|---:|---:|---:|
| shape-A `(192,256,320,128)` | 0.303280 ms | 0.256920 ms | 15.29% | 0.327100 ms | 0.268340 ms | 17.96% |
| unaligned `(191,255,319,127)` | 0.299215 ms | 0.245225 ms | 18.04% | 0.324840 ms | 0.262550 ms | 19.18% |

candidate 每轮额外做一次 single-step profiler：

- shape-A：8.24/8.54/8.44 μs，均为唯一 `triton_npu_different_k_mm_plus_mm` task。
- unaligned：13.36/13.32/13.42 μs，均为唯一融合 task。

unaligned candidate 第三轮保留一个 2.54338 ms 的孤立 host 长尾；三轮主判据和 p99 均未隐藏该样本。

结果目录：`results/t023_mmplus_different_k_integration_performance_20260821/`。

## 显存与根因

| cohort | pure output | baseline additional peak | candidate additional peak | candidate - baseline | strict gate |
|---|---:|---:|---:|---:|---|
| shape-A | 123,392 B | 246,784 B | 517,120 B | +270,336 B | 失败 |
| unaligned | 122,368 B | 244,736 B | 516,096 B | +270,336 B | 失败 |

allocator history 排除了输入 misalignment clone、Inductor planner 和泄漏：

- baseline 有两个 requested-size 122,880 B allocation，分别是第一条 mm 中间结果和最终 addmm output。
- candidate 有一个 122,880 B output，以及 Triton Ascend launcher 的 393,216 B workspace。
- 编译 metadata 为 `workspace_size=65,536 B`、grid `6×1×1`；driver 按 `workspace_size × blockNum` 分配，正好是 393,216 B。
- reserved delta 为 0；额外 allocated 是单次 launch 的真实 workspace，不是缓存扩容假象。

严格门槛定义为 candidate additional peak 不高于 baseline additional peak加一个 pure output。两个 cohort 均超过该线，不能用 15%–18% 的性能收益覆盖显存结论。

workspace 优化筛选见 [T-024 报告](t024_mmplus_different_k_workspace_20260821.md)。

## 环境限制

当前 PyTorch 是 editable source install，但其 source view 没有可供 launcher 使用的 `torch/include`；安装 headers 要求 C++20，安装 Triton Ascend launcher 固定 C++17；torch_npu installed headers 还引用当前 CANN headers 中缺少的 conditional graph 类型。无 shim 时 candidate 编译失败会安全回退到 extern，不会产生错误结果，但也没有融合收益。

本报告的 candidate 功能和性能使用进程局部、审计专用 C++20/header wrapper。该 wrapper 不在产品源码和 wheel 中。正式关闭 T-023 环境项需要在版本/header 合同匹配的独立环境重跑 fresh compile、两 cohort correctness 和 single-task smoke；无需重做已经闭环的全部 standalone 矩阵。

## 最终边界

- 保持默认关闭；只在明确接受每次 launch 约 270 KiB 额外峰值 allocated 的场景灰度启用。
- 保持 131072 output elements 上限和 static/layout/dtype gate。
- large 继续 `supported-neutral-hold`；dynamic、empty、same-K 和 arbitrary stride 继续走 fallback。
- template render、compile、autotune 或 selection 失败时保留 extern fallback。
- 不把 T-024 grouped kernel 接入产品，也不修改 Triton Ascend driver 来掩盖 workspace。
