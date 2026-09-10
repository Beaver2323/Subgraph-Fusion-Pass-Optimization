# T-102 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：5 个GPU-ready单元，0 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-102 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-1

```python
# torch/_inductor/fx_passes/fuse_attention.py:43
def _sfdp_pattern_1(...):
    # QK转置→除以缩放→softmax→V
    return explicit_matmul_softmax_attention

def _sfdp_replacement_1(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：无mask、无dropout，scale=1/inv_scale。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_1_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_1` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-2

```python
# torch/_inductor/fx_passes/fuse_attention.py:65
def _sfdp_pattern_2(...):
    # QK转置→乘缩放→softmax→V
    return explicit_matmul_softmax_attention

def _sfdp_replacement_2(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：无mask、无dropout，直接保留scale_factor。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_2_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_2` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-3

```python
# torch/_inductor/fx_passes/fuse_attention.py:87
def _sfdp_pattern_3(...):
    # 除法缩放attention并带dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_3(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：把训练态dropout概率交给SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_3_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_3` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-4

```python
# torch/_inductor/fx_passes/fuse_attention.py:109
def _sfdp_pattern_4(...):
    # 乘法缩放attention并带dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_4(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：把显式matmul/softmax/dropout链收为SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_4_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_4` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-5

```python
# torch/_inductor/fx_passes/fuse_attention.py:129
def _sfdp_pattern_5(...):
    # 带attention mask的除法缩放
    return explicit_matmul_softmax_attention

def _sfdp_replacement_5(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：mask转为Q dtype后进入SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_5_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_5` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。性能worker因此调用冻结源码 `_get_sfdp_patterns(device)`，选择同编号half-inference注册输入；它保留真实shape、stride、dtype和scalar workaround，但明确属于tracker派生微图，不是社区原生benchmark或模型端到端。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-102 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF使用隔离目标图并关闭joint_graph整轮，结论只归属于该图；若发现其他joint pass变化，性能结果作废。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
