# T-105 pattern 19：因果掩码与加性 mask 合并后的 SDPA

> 更新时间：2026-09-15 00:30 CST（UTC+08:00）。本编号登记原合同、独立功能及性能处置完成；保留 PARTIAL_ALIGNED，不代表所有精度域已修复。

## Pattern 的代码和意图

```python
# 冻结 torch/_inductor/fx_passes/fuse_attention.py:552，省略标量构造和 dropout
scores = query @ key.transpose(-2, -1)
scores = scores / inv_scale
scores = torch.where(causal_mask, scores, torch.finfo(query.dtype).min)
scores = scores + attn_mask
return scores.softmax(-1).to(value.dtype) @ value

# 同文件:573，_sfdp_replacement_19，核心节选
merged_mask = torch.where(causal_mask, attn_mask, -float('inf'))
return _scaled_dot_product_attention(query, key, value,
    attn_mask=merged_mask, dropout_p=dropout_p, scale=1.0 / inv_scale)
```

意图是把因果可见性和额外浮点偏置合为一个 mask，将显式 attention 链交给 SDPA。
这是 joint FX 改写，早于 lowering 和最终 kernel 选择；命中并不保证走融合 FA 或得到加速。

## 原社区功能与派生性能不是同一组输入

原方法 `test/inductor/test_fused_attention.py:1234 _test_sdpa_rewriter_19` 用 FP32
Q/K/V `[4,2,16,32]`、bool 因果 mask 和 FP32 加性 mask `[16,16]`，两种 scale 表达式。
GPU 本编号 reference 有效；NPU 原方法完整执行、零 skip、两次精确改写。
但 `_check_common(..., has_dropout=True, check_train=False)` 未执行输出/输入梯度数值比较，
所以“原方法通过”不能写成“原方法精度已证明”。

独立性能图从同编号 **FP32 inference 注册**派生：Q/K/V `[2,4,8,16]`，
bool 因果 mask、FP32 加性 mask 均 `[1,1,8,8]`，`inv_scale=0.66666`、dropout=0。
固定种子，三角可见性，浮点样本标准差 0.25；独立 OFF/ON 均按同设备 eager 比较，
`atol=2e-3, rtol=0.2`。该新增 oracle 只覆盖派生输入，不能回填原社区未执行的断言。

较早 half 注册的 FP16 Q/K/V + FP32 mask 在 ON replacement 追踪时报 dtype 错误。
恢复社区 FP32 域不是修好 half 域，原失败继续保留于[性能精度域缺口](../../../issues/REF-sfdp-pattern-19-native/性能精度域缺口.md)。
未强制 cast mask、改产品 gate 或更换后端；尚无 CUDA 同 half 域实测，不推断 CUDA 也失败。

## NPU OFF/ON 真正执行了什么

物理 NPU 5，`triton_experimental` 在导入前选择。OFF 只移除 19 号注册，整轮 joint 与邻接仍启用；
本编号和通用 attention counter 均 0。ON 精确 `_sfdp_pattern_19_inference` 改图 1 次。

```python
# issues/REF-sfdp-pattern-19-native/evidence/benchmark-20260915T002242+0800/
# off1/debug/torch_compile_debug/run_2026_09_15_00_22_51_979867-pid_1202519/
# torchinductor/model__0_inference_0.0/output_code.py，Runner.call 顺序简写
extern_kernels.bmm(q, kt, out=scores)
triton_unk_fused__softmax_add_div_full_matmul_where_0.run(...)
extern_kernels.bmm(probabilities, v, out=output)

# 同批 on1/debug/torch_compile_debug/run_2026_09_15_00_23_28_862926-pid_1210932/
# torchinductor/model__0_inference_0.0/output_code.py，Runner.call 顺序简写
triton_unk_fused_mul_0.run(...)             # Q 乘 sqrt(scale)
triton_unk_fused_mul_transpose_1.run(...)   # K 缩放、转置物化
extern_kernels.bmm(...)
triton_unk_fused__safe_softmax_add_full_matmul_where_2.run(...)
visible = torch.ops.aten.any.dim(...)
triton_unk_fused__safe_softmax_add_full_matmul_where_3.run(...)
extern_kernels.bmm(...)
```

两条路径都在 NPU 执行；ON 是数学 SDPA 展开，**没有 FA 调用**。
已逐读两个独立功能进程的 FX/IR，以及六份计时 `Runner.call`，未发现 CPU 搬运。
ON 多了缩放/转置物化和 safe-softmax 可见性处理；这是代码结构差异，不能未经 profiler
把全部时延回退精确分摊给某个 kernel。

## 性能结果及范围

六个新进程按 OFF1/ON1/ON2/OFF2/OFF3/ON3 串行，每臂预热 10、采样 100。
对各臂 p50/p99 再取三轮中位数，host 同步墙钟与 NPU Event 分别统计。

| 指标 | OFF(ms) | ON(ms) | 改善率 |
|---|---:|---:|---:|
| host p50 | 0.506780 | 0.701260 | -38.38% |
| host p99 | 0.558491 | 0.809817 | -45.00% |
| Event p50 | 0.390110 | 0.582570 | -49.33% |
| Event p99 | 0.409146 | 0.620413 | -51.64% |

结论 **PERF_REGRESSED / PARTIAL_ALIGNED**，不调整默认配置。
这里只是已编译推理微图，不包含编译、训练 backward、optimizer 或完整模型端到端。
保留的限制包括原例无数值 oracle、额外 half/mixed-mask 编译缺口、当前数学展开非融合 FA。

[原样本与可离线复核绑定](attention_bundles/pattern-19.json)、[性能汇总](performance_summary.json)、
[功能合同与对齐范围](functional/pattern-19.json)。

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py \
  --pattern 19 --check-current
```

复核命令本身 `device_execution=false`，表示它只验原件与重算样本，不否定原运行的真机执行。
