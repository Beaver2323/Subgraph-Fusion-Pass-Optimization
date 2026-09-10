# T-106 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：4 个GPU-ready单元，1 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-106 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-21

```python
# torch/_inductor/fx_passes/fuse_attention.py:628
def _sfdp_pattern_21(...):
    # T5加法mask与FP32 softmax
    return explicit_matmul_softmax_attention

def _sfdp_replacement_21(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：融合为无缩放SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_21_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_21` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-22

```python
# torch/_inductor/fx_passes/fuse_attention.py:654
def _sfdp_pattern_22(...):
    # T5加法mask且返回K/V
    return explicit_matmul_softmax_attention

def _sfdp_replacement_22(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：保持多输出合同。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_22_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_22` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-23

```python
# torch/_inductor/fx_passes/fuse_attention.py:684
def _sfdp_pattern_23(...):
    # T5零mask消去且返回K/V
    return explicit_matmul_softmax_attention

def _sfdp_replacement_23(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：用attn_mask=None的SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_23_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_23` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-24

```python
# torch/_inductor/fx_passes/fuse_attention.py:716
def _sfdp_pattern_24(...):
    # MBart/PLBart reshape+bmm与异dtype mask
    return explicit_matmul_softmax_attention

def _sfdp_replacement_24(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：恢复四维后以scale=1融合。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_24_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_24` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。性能worker因此调用冻结源码 `_get_sfdp_patterns(device)`，选择同编号half-inference注册输入；它保留真实shape、stride、dtype和scalar workaround，但明确属于tracker派生微图，不是社区原生benchmark或模型端到端。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-106 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF使用隔离目标图并关闭joint_graph整轮，结论只归属于该图；若发现其他joint pass变化，性能结果作废。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-fuse-attention-sfdp-pattern-25` | `deferred-explicit-cuda-disabled-xpu-only` | 注册extra_check明确disable_cuda，GPU类也只在HAS_XPU_AND_TRITON时暴露该编号；禁止绕过CUDA关闭制造reference。 |
