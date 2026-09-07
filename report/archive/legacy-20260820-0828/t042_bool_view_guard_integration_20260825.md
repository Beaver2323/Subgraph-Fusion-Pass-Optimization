# T-042：bool-cast view-chain 性能 guard 与安装态闭环

## 结论

`bool_cast_mul_to_where_pass` 现只改写包含至少一个 view/broadcast 节点的
`bool cast → view chain → mul`。direct `bool cast → mul` 保持原图，避免 T-041 已证明的
p99/device 回退；view-chain 路径代码不变，保留 p50/p99 36.30%/39.90% 的收益。

新 source-built wheel 已安装，源码态/安装态 76/76 FX 测试与 3/3 NPU fullgraph worker
通过。最终该 pass 在“整数/布尔 exact-zero dtype + 非空单用户 view chain”能力域内为
`supported-beneficial`；direct 与浮点路径安全保持原图。没有新增 Triton kernel。

## 修改与回滚边界

- `ascend_graph_pass.py` 在 `_walk_back_view_chain_to_cast` 匹配后增加 `if not chain:
  continue`。空 chain 即 direct cast，不再改写；unsqueeze/squeeze/view/reshape/expand 等
  非空 chain 继续走既有 dtype、single-user、fake-meta 和 where replay。
- 原 direct integer positive 测试改名为 direct-not-folded，并断言 cast/mul `1/1→1/1`、
  where `0→0`。view-chain 测试仍断言 cast/mul `1/1→0/0`、where `0→1`、unsqueeze
  `1→1`；浮点 Inf/NaN 测试保持。
- 回滚边界只有上述 guard、注释和 direct 断言，不涉及 PyTorch、Triton、C++ 或 vendor op。

## Wheel 与回归

- T-040 wheel 已归档为
  `artifacts/torch_npu_t040_before_t042_bool_view_perf_guard.whl`，SHA256
  `b273aeedcb9d1367de65328bea78a448ff1eb81fa4a85dca3f910e556c7b2460`。
- T-042 wheel 为
  `src/torch_npu/dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，SHA256
  `ea801e791373b0bd3adf9d4bfb6253ace75afa800c71b0451c9b206e4664fe5a`，已用
  `pip install --no-deps --force-reinstall` 安装。
- 构建与安装日志分别为 `results/t042_torch_npu_wheel_build_20260825.log`、
  `results/t042_torch_npu_wheel_install_20260825.log`。wheel archive 与 installed
  site-packages 均确认含 `if not chain` guard。
- 源码态 76/76：`results/t042_source_fx_tests_20260825.log`。
- 安装态 76/76：`results/t042_installed_fx_tests_20260825.log`。

## NPU 功能复验

物理 NPU1 运行前后无其他进程；default backend、inference、fullgraph、fresh cache，统一
使用已登记 audit shim：

| profile | 变换前后 | 结果 |
|---|---|---|
| direct int32 | cast/mul `1/1→1/1`，where `0→0` | complete，mismatch 0 |
| view-chain int32 | cast/mul `1/1→0/0`，unsqueeze `1→1`，where `0→1` | complete，mismatch 0 |
| float Inf/NaN | cast/mul `1/1→1/1`，where `0→0` | complete，NaN/signbit 合同通过 |

聚合结果为
`results/t042_bool_view_guard_compile_20260825/aggregate/aggregate.json`：3/3 complete、全部
图/输出合同通过、累计 mismatch 0。

## 性能证据继承边界

T-042 没有重测 T-041 的 24 个 worker，因为新 guard 对 view-chain candidate 代码没有任何
修改，只把 direct candidate 变成已测 baseline 图。最终行为可直接组合为：

- direct：使用 T-041 baseline，避免候选 p99 -19.02% 与 device duration 62.56→73.86 μs。
- view-chain：使用 T-041 candidate，p50/p99 0.440025/0.492800→0.280315/0.296190 ms，
  改善 36.30%/39.90%。
- 浮点：由 T-040/T-042 IEEE guard 保持原图，不进入性能正域。

这是一项 pass capability/performance gate 优化，不是新 kernel 优化。后续直接进入 B2 最后
三个复合 pass；只有模型 shape 显示新的 view-chain 边界时才扩展本 pass cohort。
