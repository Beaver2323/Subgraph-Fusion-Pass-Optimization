# T-049/T-050：B4 attention 首轮功能与性能报告

## 结论

B4 的 30 个概念 attention pattern 已进入动态审计。首轮选择的 7 个代表 family 均在
Ascend910B2 上精确触发目标 matcher、通过数值合同；其中 pattern 1 的静态 fp16 inference
路径已经完成三轮配对性能验证，结论为 `supported-beneficial`。本轮没有修改产品源码，也没有
新增手写 Triton attention kernel。

| pattern | 精确触发 | 最终 codegen | 最大绝对误差 | 当前结论 |
|---|---:|---|---:|---|
| `_sfdp_pattern_1` | 1 | `npu_fusion_attention_v3` / `aclnnFlashAttentionScore` | 0.0009765625 | `supported-beneficial`（fp16 静态 inference） |
| `_sfdp_pattern_5` | 1 | 重新展开为 2 BMM + Triton softmax/cast | 0.0009765625 | 功能通过，性能未测 |
| `_sfdp_pattern_13` | 1 | `npu_fusion_attention_v3` / `aclnnFlashAttentionScore` | 0.002197265625 | 功能通过，性能未测 |
| `_sfdp_pattern_18` | 1 | Triton logical-not + vendor attention | 0.001953125 | 功能通过，性能未测 |
| `_sfdp_pattern_21` | 1 | 重新展开为 2 BMM + Triton softmax/cast | 0.001953125 | 功能通过，性能未测 |
| `_sfdp_pattern_28` | 1 | 3 个 Triton clone + vendor attention | 0.001220703125 | 功能通过，性能未测 |
| `_sfdp_pattern_29` | 1 | 重新展开为 2 BMM + Triton softmax/cast | 0.0009765625 | 功能通过，性能未测 |

“matcher 已触发”和“最后执行 FlashAttention”不是一回事。pattern 5、21、29 的 matcher 与
`fuse_attention` counter 都是 1，但 NPU dispatcher 又把 SDPA 展开成数学路径，所以它们目前只能
证明 pass 功能正确，不能计为 vendor fusion 已落地或性能有益。

## 背景：attention pattern 在编译链中的位置

PyTorch 的 `torch/_inductor/fx_passes/fuse_attention.py` 描述 30 个概念 pattern；lazy init 会按
设备、dtype、training/inference 等生成具体注册 entry。matcher 找到 `matmul → scale/mask →
softmax/dropout → matmul` 后，用 SDPA 替换局部子图。NPU 后端随后决定 SDPA 是：

1. 映射为 `torch.ops.npu.npu_fusion_attention_v3`；
2. 保留额外 mask/clone Triton kernel，再调用 vendor attention；
3. 因当前输入合同不满足 NPU fused path，又展开回 BMM 和 Triton pointwise/reduction。

因此本批每条都同时检查四层证据：目标 pattern counter、总 `fuse_attention` counter、generated
code 和 profiler。只看一个 counter 会把 pattern 5/21/29 误报为“硬件融合成功”。

## T-049：scale 合同

`torch.ops.npu_graph.npu_fa` 的 positional 与 keyword 参数最终都经 dispatcher 按 schema
规范化，所以 `scale=0.5/1/2` 的两种调用输出完全相同。此前仅由 Python wrapper 形参写法推测
keyword 可能绕过倒数逻辑是不成立的。

动态对照确认 wrapper 中的 `scale` 沿用历史“除数”合同：输入 0.5 或 2.0 时，wrapper 与底层
vendor 同 scale 的最大差为 `0.1217041015625`，与 vendor `1/scale` 的结果误差为 0。历史实现的
pattern graph 本来执行 `.div(inv_scale_factor)`，wrapper 的倒数正是为了把这个除数转换成 vendor
乘数。因此当前不把它判为数值 bug，但参数名容易误读，后续文档/测试应明确 divisor 语义。
`scale=0` 的 positional/keyword 都抛 `ZeroDivisionError`，属于无效除数边界，暂不修改产品。

第 31 条 `npu_fusion_attention_graph` 是包装入口，不等于前述 30 个 matcher；本次完成了 scale
forward 合同，但 backward、动态 shape 和性能仍未闭环，所以矩阵不提前给最终 verdict。

## T-049：代表 pattern 功能结果

pattern 1 和 13 不需要 Triton launcher 辅助 kernel，能在当前 wheel 无 shim 完成编译与执行。
pattern 5、18、21、28、29 的初次无 shim fresh launcher 均因 editable PyTorch source include
view 缺 `ATen/ATen.h` 失败。失败图与 `output_code.py` 表明阻断点是生成的辅助 Triton launcher，
不是 matcher、SDPA replacement 或 vendor attention 本身。

为完成归因，在新 cache 中复用了 T-022 已登记的 audit-only C++20/CANN header wrapper 和
installed PyTorch headers。这个 shim 同时用于 A/B 两侧，只修复开发环境 launcher 合同，不属于
产品功能，也不能据此声称当前 fresh 环境已原生可用。有效重跑后 7/7 代表 family 的
structure/value/dtype/shape 合同都通过。

## T-050：pattern 1 配对性能

测试图为无 mask 的 `(4,8,128,64)` fp16 静态 inference attention。第一次只关闭 pattern 1 时，
等价的 `_sfdp_pattern_3_half_inference` 会接管同一个图，因此那组 smoke 数字无效。正式 baseline
同时关闭 pattern 1/3 的两个 entry；candidate 保持产品默认。这个控制变量证明的是等价 inference
family 被禁用与启用的差异，但本轮最终 verdict 只回填到已经直接覆盖的 pattern 1，不能据此关闭
pattern 3 的 training/dropout 范围。

两侧各 3 个 fresh process，顺序 `B1,C1,C2,B2,B3,C3`，每轮 warmup 10、runs 100。六轮均满足：
baseline fusion/pattern counter 为 0，candidate 为 1；输出 shape/dtype/finite 正确，最大绝对误差
均为 `0.0009765625`，`atol=rtol=0.02` 通过。

| 指标（三轮中位） | baseline：2 BMM + 2 Triton | candidate：1 FlashAttention | 变化 |
|---|---:|---:|---:|
| P50 | 0.581895 ms | 0.310165 ms | 改善 46.70% |
| P99 | 0.601770 ms | 0.335440 ms | 改善 44.26% |
| mean | 0.583300 ms | 0.312600 ms | 改善 46.41% |
| 首次 compile+run | 69,231.59 ms | 2,959.96 ms | 改善 95.72% |
| additional allocated peak | 204,477,440 B | 25,955,328 B | 减少 87.31% |
| additional reserved peak | 224,395,264 B | 27,262,976 B | 减少 87.85% |
| device task/step | 4 | 1 | 减少 75% |

P50 改善大于 10%、P99 未回退、allocated peak 未增加，三个预登记 gate 全部通过。pattern 1 在
本次 fp16 静态 inference 范围内记 `supported-beneficial`；既有 vendor kernel 已明显优于展开路径，
没有理由用手写 Triton 替代。

## 证据与失败/中性尝试

- scale：`results/t049_b4_attention_scale_20260826/result.json`
- pattern 1 无 shim：`results/t049_b4_attention_pattern1_smoke_debug_20260826/result.json`
- pattern 13 无 shim：`results/t049_b4_attention_pattern13_smoke_20260826/result.json`
- pattern 5/18/21/28/29 的有效 shim 结果：对应
  `results/t049_b4_attention_pattern*_shim_smoke_20260826/result.json`
- 同名无 `_shim_` 的失败日志保留开发环境 header 阻断；初版 pattern 29 未精确命中、初版只关闭
  pattern 1 的性能 smoke 也保留，但不进入结论。
- 正式性能聚合：
  `results/t050_b4_attention_pattern1_performance_20260826/aggregate/aggregate.json`

## 下一步

1. pattern 13 的第二条 paired 已由 T-051 完成：P50 只改善 0.99%，但 task/编译/内存显著改善，
   记 resource-beneficial；见 `t051_b4_attention_pattern13_performance_20260826.md`。
2. 对 pattern 5/21/29 先区分 NPU dispatcher 主动 re-expand 的 capability 原因，再决定应扩大
   vendor attention gate、保留数学 fallback，还是优化辅助 kernel；不能看到 Triton 就重写 attention。
3. pattern 18/28 已落到 vendor attention，但辅助 logical-not/clone 可能主导收益，后续需用同样的
   单 pass 隔离和 profiler 判断是否值得融合这些辅助操作。
4. 扩展剩余 23 个概念 pattern，并补 training/dropout、mask dtype/broadcast、dynamic shape、
   backward；第 31 条 wrapper 继续补 backward 与文档合同。
