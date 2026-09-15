# T-102 pattern 3：功能、修复与性能对照

> 更新时间：2026-09-15 20:15 CST（UTC+08:00）。原社区合同、安装修复、精确OFF/ON与性能处置完成；不是产品PR已合入。

## Pattern 意图与代码

softmax(QKᵀ / scale) → dropout → V：识别带随机丢弃的训练 attention。 匹配/替换位于 joint FX 阶段，在 lowering、scheduler、autotune 和最终 kernel 选择之前。

```python
# PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:87
def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)

def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )
```

## 功能测例及 GPU/NPU 对比

原入口为 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_3_gpu`。
GPU使用用户已回传的冻结原方法/只读精确观察包，本轮未重新跑GPU。NPU仅设备及代码符号适配，不更改数值阈值。

| 项目 | GPU reference | NPU experimental |
|---|---|---|
| 原方法与本编号 | 原生通过，有目标边界FX | 修复前训练未命中；新安装态1方法、0skip、本编号2次改写 |
| 输出/梯度 | 以原方法实际断言为界 | 原方法没有随机输出/梯度数值 oracle。当前仅原执行/改写合同通过，低非零 dropout 性能微图的数值验证另列。 |
| 修复层 | 不修改CUDA或上游 | 活动NPU decomposition重建1～5训练pattern，原replacement/guard保留 |
| 当前统计 | [GPU审查](gpu_reference_review.json) | NEWLY_SUPPORTED；PARTIAL_ALIGNED；性能PERF_IMPROVED |

推理时 dropout=0 由相邻无 dropout 编号承载；本编号只统计训练的两次改写。 对齐标签限本次冻结合同，FULL_ALIGNED不代表所有dtype/shape或代码生成逐指令一致；FX改写不自动等于融合FA或性能收益。

[完整修复说明与调用栈](../../../issues/REF-sfdp-pattern-3-native/训练注册修复报告.md)包含实际36行部署代码、五原例和六正负边界、前后原件。
[部署证明](../../../issues/REF-sfdp-pattern-1-native/deployment-20260915-training/deployment.json)验签对应两个安装文件与新进程。

## 性能测例来源、输入与范围

没有将已融合SDPA API benchmark冒充本FX rewrite benchmark。复用冻结 `_get_sfdp_patterns` 同编号half training的search、shape、stride和dtype，固定种子normal std0.25替代未初始化empty；保留requires_grad。
3/4使用社区极低非零dropout设计1e-11，不能置0后测邻接；1/2/5无dropout。

| 输入 | shape/标量 | dtype | requires_grad |
|---|---|---|---|
| query | [2, 4, 8, 16] | torch.float16 | True |
| key | [2, 4, 8, 16] | torch.float16 | True |
| value | [2, 4, 8, 16] | torch.float16 | True |
| inv_scale_factor | 2.0 | float | — |



仅training-forward微图；不含backward、首次编译或完整模型。每臂与同卡eager比较，派生域rtol=0.2、atol=2e-3，不回写原社区方法的容差；OFF只删本编号，不关闭整轮joint或邻接。合法OFF的attention计数为0，ON本编号改写>=1。
1号历史推理OFF被3号接替，故改测与本次修复一致的training合同；[失败与修订记录](../../../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)保留，不借用T-106的特许豁免。

## 实际生成代码对照

下列摘取实际OFF、ON调用行，完整FX/IR/代码位于注释路径；Triton softmax/转换核与缓存分配未全展开，不是完整可执行脚本。

```python
# issues/REF-sfdp-pattern-3-native/evidence/functional-20260915T194634+0800/off/debug/torch_compile_debug/run_2026_09_15_19_48_09_014528-pid_1821165/torchinductor/model__10_forward_31.0/output_code.py
extern_kernels.bmm(reinterpret_tensor(primals_2, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(primals_1, (8, 16, 8), (128, 1, 16), 0), out=buf0)
buf7 = torch.ops.npu._npu_dropout.default(buf6, 1e-11)
extern_kernels.bmm(reinterpret_tensor(buf8, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(primals_3, (8, 8, 16), (128, 16, 1), 0), out=buf10)
```
```python
# issues/REF-sfdp-pattern-3-native/evidence/functional-20260915T194634+0800/on/debug/torch_compile_debug/run_2026_09_15_19_48_59_777678-pid_1833920/torchinductor/model__10_forward_34.0/output_code.py
buf0 = torch.ops.npu.npu_fusion_attention_v3.default(primals_2, primals_1, primals_3, 4, 'BNSD', None, None, None, 0.5, 0.99999999999, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
```

## 性能结果

已完成独立功能门禁和六臂计时，结论 **PERF_IMPROVED**。正改善率表示时延下降，负值表示回退。

| 时钟/分位 | OFF ms | ON ms | 改善率 |
|---|---:|---:|---:|
| host_ms p50 | 0.935295 | 0.542540 | +41.99% |
| host_ms p99 | 1.098436 | 0.600027 | +45.37% |
| event_ms p50 | 0.802690 | 0.418000 | +47.93% |
| event_ms p99 | 0.879012 | 0.464326 | +47.18% |

同 NPU 5、六个新进程，顺序 OFF1/ON1/ON2/OFF2/OFF3/ON3；每臂预热10次、host及Event各100样本。汇总取三轮各臂p50/p99的中位数，原样本、波动和compile_ms保留。compile_ms起点位于先行lazy_init之后，不是包含训练注册trace的完整冷启动成本。[功能门禁](performance_gates/pattern-3-executed.json)、[完整验签包](attention_bundles/pattern-3.json)、[时延原始记录](performance_summary.json)。

## 交付边界

原方法未比较的随机梯度、BF16与其他shape不外推；低dropout微图通过不能补写原高dropout数值oracle。局部收益不代表整个训练或默认开关决策。未改产品默认配置；本地产品提交3a0b8179d8c6db72ae1c86ea02e88ea5229556ef尚未推送产品仓/运行社区CI或发布wheel。
