# T-105 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：5 个GPU-ready单元，0 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-105 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-16

```python
# torch/_inductor/fx_passes/fuse_attention.py:421
def _sfdp_pattern_16(...):
    # BERT Large mask+dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_16(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：CUDA命中后故意保留数学路径保证数值；非CUDA可转SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_16_inference_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_16` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-17

```python
# torch/_inductor/fx_passes/fuse_attention.py:464
def _sfdp_pattern_17(...):
    # DistilBERT masked_fill+dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_17(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：规范布尔mask并保留dropout。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_17_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_17` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-18

```python
# torch/_inductor/fx_passes/fuse_attention.py:504
def _sfdp_pattern_18(...):
    # GPT2 causal mask并返回K/V
    return explicit_matmul_softmax_attention

def _sfdp_replacement_18(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：融合attention同时保持多输出K/V。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_18_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_18` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-19

```python
# torch/_inductor/fx_passes/fuse_attention.py:552
def _sfdp_pattern_19(...):
    # GPT2 causal mask叠加attention mask
    return explicit_matmul_softmax_attention

def _sfdp_replacement_19(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：先合成mask再交给SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_19_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_19` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-20

```python
# torch/_inductor/fx_passes/fuse_attention.py:588
def _sfdp_pattern_20(...):
    # 新版DistilBERT先缩放Q的mask+dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_20(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：覆盖不同缩放位置。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_20_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_20` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。性能worker因此调用冻结源码 `_get_sfdp_patterns(device)`，选择同编号half-inference注册输入；它保留真实shape、stride、dtype和scalar workaround，但明确属于tracker派生微图，不是社区原生benchmark或模型端到端。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-105 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF使用隔离目标图并关闭joint_graph整轮，结论只归属于该图；若发现其他joint pass变化，性能结果作废。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
