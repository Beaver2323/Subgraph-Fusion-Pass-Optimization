# T-106 pattern 24：reshape/BMM attention 改写生效，NPU数学路径性能回退

> 更新时间：2026-09-14 23:57 CST（UTC+08:00）。原合同、独立功能及六臂计时完成；结论PARTIAL_ALIGNED / PERF_REGRESSED，未修改默认开关。

## 1. Pattern 的意图

```python
# 冻结PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:716
# _sfdp_pattern_24语义节选，省略部分reshape。
q, k, v = (x.reshape(bs * n_head, -1, head_size) for x in (query, key, value))
weights = torch.bmm(q, k.transpose(1, 2))
weights = weights.view(bs, n_head, seq_len, -1) + attention_mask
probabilities = weights.reshape(bs * n_head, seq_len, -1).softmax(-1)
if query.dtype == torch.half:
    probabilities = probabilities.to(torch.half)
return torch.bmm(probabilities, v).view(bs, n_head, seq_len, head_size)

# 同文件:740，replacement本身已有mask dtype转换，不是tracker新增的修复。
return _scaled_dot_product_attention(
    query, key, value,
    attn_mask=attention_mask.to(dtype=query.dtype), is_causal=False, scale=1,
)
```

它识别MBart/PLBart样式的reshape+BMM链及异dtype加性mask，恢复四维接口并改写为SDPA。
原图没有缩放，因此scale=1。joint阶段改写位于lowering、scheduler和kernel选择之前；
改写为SDPA不代表NPU必然选择融合FA内核。

## 2. 功能和性能分别覆盖什么

| 合同 | 输入/数值检查 | 结果 |
|---|---|---|
| 原社区 `_test_sdpa_rewriter_24` | FP32 Q/K/V `[4,2,16,32]`，FP32 mask `[1,1,16,16]`，原`atol=1e-3, rtol=0.2` | 1/1零skip、1次Tensor比较、1次本编号改写 |
| 注册样例派生性能图 | FP16 Q/K/V `[2,4,8,16]`，FP32 mask `[1,1,8,8]`，固定种子正态初始化 | 独立OFF/ON均对NPU eager通过；派生`atol=2e-3, rtol=0.2` |
| 六臂计时 | 同一性能图，compiled forward，无dropout | 完成；不包含反向或模型端到端 |

社区方法位于`test/inductor/test_fused_attention.py:1457`。派生FP16/FP32混合域不能冒充原例FP32输入；
两者数值门禁、原件和容差分列。GPU/NPU比较按各自同设备eager，不要求跨设备逐位一致。

## 3. GPU/NPU行为及真实生成代码

GPU原例具名确认24号并使用CUDA SDPA路径；NPU同编号改写后仍走SDPA数学展开。
最初NPU适配器只接受fusion-attention符号而误拦截数学路径；保留原失败，审查实际FX/IR/codegen后，
将断言限定为“精确本编号改图+两次NPU BMM+safe-softmax+真实NPU执行”。原数值断言未修改。
见[代码断言适配](../../../issues/REF-sfdp-pattern-24-native/代码断言适配分析.md)。

```text
原社区方法 / 派生search_fn
 → Dynamo/AOT → joint matcher 24号replacement
 → SDPA decomposition / NPU lowering
 → FP32 BMM + safe-softmax + FP32 BMM
 → 输出cast回FP16（仅派生half图） → NPU eager数值断言
```

```python
# issues/REF-sfdp-pattern-24-native/evidence/benchmark-20260914T234441+0800/
# off1/debug/torch_compile_debug/run_2026_09_14_23_50_22_340681-pid_751249/
# torchinductor/model__0_inference_0.0/output_code.py，Runner.call语义节选。
extern_kernels.bmm(q_fp16, kt_fp16, out=scores_fp16)
# 单个Triton kernel：加FP32 mask、softmax、概率转FP16。
extern_kernels.bmm(probabilities_fp16, v_fp16, out=out_fp16)

# 同目录on1/debug/torch_compile_debug/run_2026_09_14_23_51_00_157821-pid_760041/
# torchinductor/model__0_inference_0.0/output_code.py，语义节选。
# Q/K提升FP32，转置；V稍后也提升FP32。
extern_kernels.bmm(q_fp32, kt_fp32, out=scores_fp32)
# mask经replacement转FP16；Triton判断非-inf。
not_all_masked = torch.ops.aten.any.dim(non_inf, -1, True)
# Triton：safe-softmax，处理全遮挡行。
extern_kernels.bmm(probabilities_fp32, v_fp32, out=out_fp32)
# 最后Triton输出cast回FP16。
```

六份实际Runner.call都已读取。三OFF均2个FP16 BMM+1个softmax kernel；三ON均2个FP32 BMM、
额外cast/转置、safe-softmax与NPU `aten.any`。没有fusion-attention调用，也没有CPU搬运。
`aten.any`是图内NPU调用，不是CPU fallback。
OFF只删除24号entry，保留整轮joint和其他编号，且没有其他attention接替；本编号OFF0/ON1。

## 4. 六臂性能与处置

物理NPU5、固定`triton_experimental`、新进程OFF1/ON1/ON2/OFF2/OFF3/ON3串行。
每臂预热10次、host/Event各100样本，取三轮p50/p99中位数；编译耗时和峰值内存另存原件。

| 指标 | OFF(ms) | ON(ms) | 改善率 |
|---|---:|---:|---:|
| host p50 | 0.489360 | 0.745515 | -52.34% |
| host p99 | 0.545170 | 0.810954 | -48.75% |
| Event p50 | 0.373470 | 0.622810 | -66.76% |
| Event p99 | 0.403437 | 0.663223 | -64.39% |

结论`PERF_REGRESSED`，不是“未命中”。额外转换、FP32计算和安全softmax处理提供了可追查的开销线索，
但没有独立profiling实验把全部回退归给某一个算子。保持`PARTIAL_ALIGNED`明确GPU/NPU最终实现差异。
本轮不擅改产品开关；后续若评审目标级关闭或math优化，要另存变更前后证据，不覆盖此次实际回退。

[完整归档与哈希绑定](attention_bundles/pattern-24.json)、[功能比较及部分对齐范围](functional/pattern-24.json)、
[六臂性能原样本](performance_summary.json)、[最小适配报告](../../../issues/REF-sfdp-pattern-24-native/适配报告.md)。

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py \
  --pattern 24 --check-current
```

这是仓库文本的离线重算入口，不重新运行NPU；正式完成仅覆盖此编号，不包含21/22。
