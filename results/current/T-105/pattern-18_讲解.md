# T-105 pattern 18：布尔掩码 attention，同时保留 K/V 输出

> 更新时间：2026-09-15 00:25 CST（UTC+08:00）。本编号原合同、功能对照及性能处置完成；不是整批T-105完成。

## Pattern 和意图

```python
# 冻结PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:504
# 主体语义节选，省略full标量构造和dropout。
query, key, value = (x.transpose(1, 2) for x in (query, key, value))
scores = query @ key.transpose(-2, -1)
scores = torch.where(causal_mask, scores / inv_scale, torch.finfo(query.dtype).min)
return scores.softmax(-1) @ value, key, value

# 同文件 _sfdp_replacement_18：语义节选。
return (
    _scaled_dot_product_attention(q, k, v, attn_mask=causal_mask,
                                  scale=1.0 / inv_scale, dropout_p=dropout_p),
    k, v,
)
```

意图是把“分数矩阵→缩放→掩码→softmax→乘V”交给SDPA，且仍返回转置后的K/V，不能只验证第一个输出。
布尔mask的True表示保留；NPU融合API使用相反含义，因此生成图还要expand和logical_not。
joint阶段的FX替换先于lowering和kernel选择；同编号命中和最终生成融合kernel是两项不同证据。

## 功能与性能的输入边界

社区来源为`test/inductor/test_fused_attention.py:1122`的`_test_sdpa_rewriter_18`。
原方法保留batch=4和1、两种缩放表达方式，FP32输入；四次精确改写，四组三元组共12个Tensor叶子比较。
注意原社区方法在返回前又将 K/V `permute([0,2,1,3])`，最终 K/V 恢复输入布局；
上方注册 pattern 及派生微图返回的是 attention 布局 K/V。两种返回合同均按各自原输入对照，不能混写成相同的最终 shape。
每个叶子都在NPU上与同设备eager比较，原容差为`rtol=0.2`，`atol=1e-3`或原方法声明的`2e-3`。
较早观察器只统计顶层Tensor造成的“0次数值比较”已经纠正并重跑，旧件未覆盖。

性能图来自同编号注册样例：FP16连续Q/K/V各`[2,4,8,16]`，布尔mask`[2,1,1,4]`，
固定种子初始化、三角mask，`inv_scale=0.66666`、dropout=0。它与原方法的规模和dtype不同，
独立声明`atol=2e-3, rtol=0.2`并另跑OFF/ON数值门禁；不能用派生图回填原例断言。
这里只测compiled forward，不含反向、编译时间或模型端到端。

## GPU/NPU及目标级OFF/ON

| 路径 | 行为 |
|---|---|
| GPU原社区方法 | 本编号reference已通过；GPU产物和具名改写记录见gpu_reference_review.json |
| NPU原方法 | 完整方法通过，4次本编号改写、12次Tensor比较 |
| NPU派生OFF | 只移除18号注册，保留整轮joint和其他编号；本编号及通用attention计数均0 |
| NPU派生ON | 18号一次精确改写；最终为掩码处理加NPU融合attention，输出及K/V视图保留 |

```python
# 原件：issues/REF-sfdp-pattern-18-native/evidence/
# benchmark-20260914T233705+0800/off1/debug/torch_compile_debug/
# run_2026_09_14_23_37_14_508262-pid_563622/torchinductor/model__0_inference_0.0/output_code.py
# Runner.call语义节选；省略分配、转置及softmax子kernel参数。
extern_kernels.bmm(q, kt, out=scores)
# Triton：缩放、where、amax、exp、sum、归一化
extern_kernels.bmm(probabilities, v, out=output)

# 同一benchmark目录/on1/debug/torch_compile_debug/
# run_2026_09_14_23_38_57_854224-pid_588838/torchinductor/model__0_inference_0.0/output_code.py:154
# 省略无关默认参数；Q/K/V为[2,8,4,16]视图。
buf1 = torch.ops.npu.npu_fusion_attention_v3.default(
    q, k, v, 8, 'BNSD', None, None, inverted_mask,
    1.5000150001500014, 1.0,
)
```

已人工读取六份计时Runner.call，三OFF/三ON分别保持上述实现，无CPU搬运；额外有两个独立功能进程。
各进程绑定同一冻结源码、安装态来源和归档worker/helper快照。后续编辑通用worker不改变本轮实际执行版本。

## 性能结果

物理NPU5，`triton_experimental`，六个独立进程按OFF1/ON1/ON2/OFF2/OFF3/ON3串行。
每臂预热10次、采样100次；host和NPU Event分别统计，取三轮分位数中位数。

| 指标 | OFF(ms) | ON(ms) | 改善率 |
|---|---:|---:|---:|
| host p50 | 0.780825 | 0.576340 | 26.19% |
| host p99 | 0.827426 | 0.755679 | 8.67% |
| Event p50 | 0.663840 | 0.453080 | 31.75% |
| Event p99 | 0.688863 | 0.505970 | 26.55% |

结论`PERF_IMPROVED`，不调整默认配置，不外推其他dtype/长度或模型收益。
原例只需设备/代码断言适配，未部署产品修复；性能入口的局部`dont_skip_tracing`仅使社区内部search_fn可被追踪。

复核入口：[完整证据绑定](attention_bundles/pattern-18.json)、[性能原样本和汇总](performance_summary.json)、
[适配过程](../../../issues/REF-sfdp-pattern-18-native/适配报告.md)、
[容器数值观察器修正](../../../issues/REF-sfdp-pattern-18-native/数值观察器修正说明.md)。

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py \
  --pattern 18 --check-current
```

离线复核`device_execution=false`只说明该复核命令不启动设备，不否认归档原件中的真机执行。
