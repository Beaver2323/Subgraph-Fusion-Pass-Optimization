# T-106 pattern 23：零加性 mask 消除后的多输出 attention

> 更新时间：2026-09-14 23:52 CST（UTC+08:00）。本编号功能/性能处置完成；21/22及整批状态另列。

## 原理与社区合同

```python
# 冻结PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:684
# _sfdp_pattern_23语义节选，省略重复的dtype转换。
q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
score = q @ k.transpose(-2, -1)
return score.float().softmax(-1).type_as(q) @ v, k, v

# 同文件 _sfdp_replacement_23：节选。
return _scaled_dot_product_attention(q, k, v, attn_mask=None, scale=1.0), k, v
```

它承接T5样式attention中常量零加性mask已经被消除的图；零mask不是“全部遮挡”。
改写不凭空添加缩放，显式`scale=1`，且保留返回的K/V。joint图改写先于SDPA lowering、调度和实际kernel选择。

社区原方法为`test/inductor/test_fused_attention.py:1428`的`_test_sdpa_rewriter_23`：
FP32三输入，形状分别覆盖`[4,2,16,32]`和`[1,2,16,32]`，原方法内部构造全零mask。
NPU原方法1/1、零skip，两次本编号精确改写，两个三元组共6次Tensor叶子比较；原`atol=1e-3, rtol=0.2`。
不包含训练或输入梯度。

性能图为注册样例派生的FP16三输入`[2,4,8,16]`，连续布局、固定种子正态std=0.25，无运行时mask、无dropout。
不同于原方法的dtype/规模，必须单独比较本机NPU eager；两臂执行前已声明`atol=2e-3, rtol=0.2`。
该测例不是社区现成benchmark，也不是完整T5模型；社区SDPA benchmark本身从已融合调用开始，不能隔离此FX改写。

## GPU/NPU及两臂真实代码

GPU原例已经具名确认23号reference；NPU也命中23号，安装态原例输出合同通过。
派生OFF只删除23号注册，其余joint优化保留；通用attention和精确目标计数都为0。
派生ON本编号一次改写，最终调用NPU融合attention。三轮两臂均检查真实Runner.call，无CPU搬运。

```python
# issues/REF-sfdp-pattern-23-native/evidence/benchmark-20260914T234441+0800/
# off1/debug/torch_compile_debug/run_2026_09_14_23_44_51_033824-pid_672258/
# torchinductor/model__0_inference_0.0/output_code.py，Runner.call语义节选。
extern_kernels.bmm(q, kt, out=score)
# Triton：FP16分数转FP32，softmax，概率转回FP16；包含转置复制kernel。
extern_kernels.bmm(probabilities, v, out=output)

# 同目录on1/debug/torch_compile_debug/run_2026_09_14_23_46_11_990046-pid_693948/
# torchinductor/model__0_inference_0.0/output_code.py，省略默认参数和具体视图。
result = torch.ops.npu.npu_fusion_attention_v3.default(
    q, k, v, 8, 'BNSD', atten_mask=None, scale=1.0, keep_prob=1.0,
)
return result[0], k, v
```

本例不需要产品修复；设备发现/代码符号断言适配保留原方法和数值容差。
性能入口对内部search_fn使用局部`dont_skip_tracing`，不关闭fullgraph、不改目标guard。
本轮执行原功能gate绑定的worker和helper归档快照，而非后续修改的公共脚本。

## 性能

物理NPU5，固定`triton_experimental`；OFF1→ON1→ON2→OFF2→OFF3→ON3六个独立进程串行。
每臂10次预热、100次host/Event采样；分别取三轮p50/p99中位数，编译耗时单列，不混入样本。

| 指标 | OFF(ms) | ON(ms) | 改善率 |
|---|---:|---:|---:|
| host p50 | 0.716410 | 0.508485 | 29.02% |
| host p99 | 0.784389 | 0.568868 | 27.48% |
| Event p50 | 0.603710 | 0.383360 | 36.50% |
| Event p99 | 0.632619 | 0.414459 | 34.49% |

结论`PERF_IMPROVED`，仅限当前派生微图；不等于其他dtype、长序列、反向或模型端到端收益，不改变默认配置。

[绑定原件及离线重算入口](attention_bundles/pattern-23.json)、[功能记录](functional/pattern-23.json)、
[六臂性能汇总与原样本链接](performance_summary.json)、[最小适配报告](../../../issues/REF-sfdp-pattern-23-native/适配报告.md)。

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py \
  --pattern 23 --check-current
```

命令只读取归档，无需复核机器再次运行NPU；原方法和八个功能/计时进程的证据仍可逐项追溯。
