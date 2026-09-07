# T-054：P-013 pattern 5 NPU 性能门禁报告

## 结论

P-013 已完成源码、wheel、安装态功能、三轮 paired 性能和邻近回归闭环。torch_npu 现在只对
NPU default backend 的 `_sfdp_pattern_5_half_inference` 关闭通用 SDPA rewrite，使 arbitrary
float-mask 图保留为 Inductor 原始数学路径。guard 相对旧 rewrite 的 P50 改善 50.28%，所有门槛
通过；这是已落地的 pass 性能优化，不需要手写完整 Triton attention。

## 修改范围

- `torch_npu/_inductor/fx_passes/joint_graph.py`：在 attention lazy registration 前，对唯一 exact
  NPU key 包装 `extra_check=False`；patch 幂等并保存原 generator/check。
- `torch_npu/_inductor/__init__.py`：default Triton backend 加载时调用该 patch。
- `test/_inductor/test_attention_pattern_gate.py`：验证 exact NPU key、非 NPU、training、pattern 21
  和重复 patch 合同，2/2 通过。

没有修改 PyTorch 通用源码、op-plugin dispatcher、schema、C++ 或 Triton kernel。旧 P-012 wheel
保存在 `artifacts/torch_npu_t053_before_p013.whl`，SHA256
`61b0031cbb027548f60745dcf0a2484503a360347dec6bd3cc2f3f2bc823ebca`。新 wheel SHA256
为 `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989`，1318 个唯一
entry，已 `--no-deps --force-reinstall` 安装。

## 安装态 A/B

同一新 wheel 内，B 侧在测试进程恢复保存的旧 pattern generator，C 侧使用默认 guard；执行顺序
`B1,C1,C2,B2,B3,C3`，warmup 10、runs 100。

| 指标（三轮中位） | 旧 rewrite | 新 guard | 改善 |
|---|---:|---:|---:|
| P50 | 0.745200 ms | 0.370545 ms | 50.28% |
| P99 | 0.767770 ms | 0.581510 ms | 24.26% |
| mean | 0.746317 ms | 0.382204 ms | 48.79% |
| 首次 compile+run | 98,470.86 ms | 41,546.24 ms | 57.81% |
| additional allocated peak | 205,527,552 B | 204,472,832 B | 减少 1,054,720 B |
| device task/step | 8 | 3 | 减少 62.5% |

两侧最大绝对误差均为 `0.0029296875`，shape/dtype/finite 与 `atol=rtol=0.02` 全部通过。
旧 rewrite exact/总 counter 为 1，新 guard为 0；两侧都没有 vendor attention。

## 邻近回归与最终分类

pattern 1、13、21 fresh regression 均 exact/总 counter 1 且数值通过；1/13 保持 vendor
FlashAttention，21 保持原 math fallback。P-013 没有扩大到未测的 training、fp32、21/29。

guard 方案本身为 `supported-beneficial`，但 `_sfdp_pattern_5_half_inference` rewrite 已被性能证据
否决并在 NPU 停用，因此矩阵最终记 `supported-pass-disabled-performance-rejected`。后续优先对
21/29 各自做 paired；不能从 pattern 5 直接外推，也不能把 float bias 粗暴转 bool。

正式 aggregate 位于
`results/t054_b4_attention_pattern5_guard_performance_20260826/aggregate/aggregate.json`。

## 中性方法记录

- 第一次 build 命令在 source `env.sh` 前设置 `set -e`，因此在构建启动前退出；旧 wheel 未被
  覆盖。调整顺序后的有效 build 才生成上述新 wheel。
- 第一组 installed smoke 在 backend loader 导入前恢复 generator，得到 `restored=0`，实际 B/C
  都是 guard 图，目录保留但不计性能。retry 先显式加载 backend，再恢复保存的原 generator，
  得到 `restored=1`、counter 1 和 8 tasks 后才运行正式六轮。
- 邻近回归第一次把 case 1 传给不支持该 choice 的代表脚本，在 argparse 阶段退出、未编译；
  随后使用独立 pattern 1 脚本有效通过。
