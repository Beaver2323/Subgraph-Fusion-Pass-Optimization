# T-104 pattern 13：从两次 BMM 到 SDPA，功能通过但微图性能回退

> 更新时间：2026-09-14 23:22 CST（UTC+08:00）。本编号功能、比较、性能处置完成；不代表T-104整批完成。

## 1. Pattern 要做什么

```python
# 冻结PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:341
def _sfdp_pattern_13(query, key, value, dropout_p):
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    return torch.bmm(attn_weight, value)

# 同文件:347，replacement主体节选；计数递增省略。
return _scaled_dot_product_attention(
    query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
    dropout_p=dropout_p, scale=1.0,
).squeeze(0)
```

三维批矩阵相乘得到注意力分数，softmax归一化后再乘V。改写把这条链交给SDPA，
以便设备选择融合实现。原图没有`1/sqrt(d)`，所以必须显式`scale=1.0`；补/去维度保留原三维接口。
这是joint阶段FX替换，发生在设备lowering、调度和kernel autotune之前；命中不保证最终性能一定提高。

```text
原社区方法/派生search_fn
  → Dynamo → AOT/Inductor joint_graph
  → 本编号PatternMatcher replacement（精确观察改写前后）
  → SDPA设备实现 → lowering → scheduler/codegen → 真实NPU执行
  → 同设备eager数值比较 → 独立新进程计时
```

## 2. 功能测例与性能测例不是同一个规模

| 合同 | 输入与比较 | 本轮结果/边界 |
|---|---|---|
| 社区原方法 `_test_sdpa_rewriter_13` | FP16，Q/K/V各`[4,8,16]`；原`atol=rtol=1e-2` | GPU、安装态NPU均1/1、零skip；NPU一次Tensor比较和一次本编号改图 |
| 注册输入派生微图 | FP16，三份连续`[1024,128,128]`；固定种子初始化；dropout=0 | 独立OFF和ON均对本机NPU eager通过，派生容差`atol=2e-3, rtol=0.2` |
| 性能 | 同一个派生微图和数值门禁，compiled forward | 不含反向、编译时间或模型其他层；不是模型端到端 |

原方法位于`test/inductor/test_fused_attention.py:885`，GPU partialmethod固定half。
虽然方法体写`p=0.5`，`check_train=False`使实际`training=False`，因此本合同dropout关闭。
派生规模来自`fuse_attention.py`注册样例，不冒充社区独立性能基准；派生容差是执行前声明的另一合同，
不能用其替代社区原例较严格的相对误差要求，也不宣称逐位一致。

## 3. GPU/NPU行为以及OFF/ON归因

| 路径 | 精确目标 | 实际输出代码 |
|---|---|---|
| GPU原例 | `_sfdp_pattern_13_half_inference`一次 | ATen Flash Attention，scale=1 |
| NPU原例/派生ON | 同编号一次，图确实变化 | `npu_fusion_attention_v3`，BNSD，scale=1，keep_prob=1 |
| NPU派生OFF | 本编号0、通用fuse_attention计数0 | 两个NPU extern BMM，加一个Triton softmax kernel |

OFF只删除本编号两个inference注册，保留132个其他条目和整轮joint pass。没有邻接attention接替OFF。
三类GPU/NPU/OFF不能互换；最终的性能对照是**同NPU、同triton_experimental后端**OFF/ON。

```python
# 本目录相对仓库根：issues/REF-sfdp-pattern-13-native/evidence/
# functional-20260914T230539+0800/off/debug/torch_compile_debug/
# run_2026_09_14_23_05_48_083077-pid_104638/torchinductor/model__0_inference_0.0/output_code.py:174
# 节选，省略视图/分配细节。
extern_kernels.bmm(q, transposed_k, out=buf0)
triton_unk_fused__softmax_0.run(buf3, 131072, 128, stream=raw_stream0)
extern_kernels.bmm(buf3, v, out=buf4)

# 同一functional目录/on/debug/torch_compile_debug/
# run_2026_09_14_23_07_36_924670-pid_130469/torchinductor/model__0_inference_0.0/output_code.py:60
# 三维注册输入补维为[1,1024,128,128]，省略其余默认参数。
buf0 = torch.ops.npu.npu_fusion_attention_v3.default(
    q4d, k4d, v4d, 1024, 'BNSD', scale=1.0, keep_prob=1.0,
)
```

已逐图检查两个功能臂和六个计时臂的`Runner.call`，均实际执行NPU路径，没有CPU转移。
不能仅凭文件顶部通用CPU分配别名推断fallback，也不能把静态字符串扫描当人工审查的替代。

## 4. 性能如何测、结果是什么

`runners/t102_t107_attention_performance_worker.py:338`：每臂10次预热、100个host/Event同步样本。
六个独立进程按OFF1→ON1→ON2→OFF2→OFF3→ON3串行，物理NPU5，使用tracker协作独占锁。
每臂重做目标/数值/来源检查；每个模式取三轮p50/p99的中位数，改善率为`(OFF-ON)/OFF`。

| 时钟/分位 | OFF(ms) | ON(ms) | 改善率 |
|---|---:|---:|---:|
| host p50 | 0.452540 | 0.483180 | -6.77% |
| host p99 | 0.493543 | 0.530825 | -7.55% |
| NPU Event p50 | 0.338830 | 0.357830 | -5.61% |
| NPU Event p99 | 0.365349 | 0.380515 | -4.15% |

结论`PERF_REGRESSED`。融合调用更少并不保证此规模更快；目前证据证明该微图回退，
尚无profile来归因具体内核瓶颈，不据此自动关闭产品默认路径，也不推广到其他shape或模型。
首编译时间另记（功能OFF约93.36秒、ON约13.90秒），没有混入上述稳态计时。

## 5. 最小适配和复核入口

原社区入口只适配设备、测试类门禁和生成代码符号断言，保留函数体、原输入、原数值阈值。
派生执行器最初直接compile内部search函数被Dynamo `MOD_SKIPLIST`阻断，8项失败分别归档。
随后局部使用冻结社区提供的`torch._dynamo.dont_skip_tracing(search_fn)`；不全局关闭skip规则，
不改产品guard，继续`fullgraph=True`。这属于测量入口适配，不是NPU产品精度修复。

- [原社区GPU/NPU代码与适配报告](../../../issues/REF-sfdp-pattern-13-native/GPU与NPU代码对照.md)
- [功能原件、FX/IR/codegen和执行器快照](../../../issues/REF-sfdp-pattern-13-native/evidence/functional-20260914T230539+0800/)
- [六臂原始样本及生成代码](../../../issues/REF-sfdp-pattern-13-native/evidence/benchmark-20260914T230901+0800/)
- [可重算证据绑定](attention_bundles/pattern-13.json)、[性能汇总](performance_summary.json)

```bash
# 从项目根执行只读离线复核；不需要原机器tmp或NPU。
python scripts/review_attention_completion.py --check-current
```

该命令的`device_execution=false`表示复核器不再启动设备，不否定归档中的真实NPU运行。
