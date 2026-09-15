# T-104 功能与性能测例讲解

> 更新时间：2026-09-14 23:21 CST（UTC+08:00）
> 状态：5/5 GPU精确编号确认；13/15号NPU原合同通过；11/12/14训练匹配失败。14推理数值已通过，不再是符号断言阻断。13号已完成独立OFF/ON及六臂性能，PERF_REGRESSED，正式1/5闭环；15尚未签性能门禁。

[13号结果与功能/性能测例逐层讲解](../results/current/T-104/pattern-13_讲解.md)解释输入规模、容差、真实内核、计时方法和回退边界。

学习入口：[13号真实GPU/NPU代码对照](../issues/REF-sfdp-pattern-13-native/GPU与NPU代码对照.md)、
[14号数学展开与代码断言适配](../issues/REF-sfdp-pattern-14-native/代码断言适配分析.md)。
每例的实际运行目录和失败栈均在对应issue的`安装态基线复核.md`，不把准备脚本当设备通过。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-104 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-11

```python
# torch/_inductor/fx_passes/fuse_attention.py:297
def _sfdp_pattern_11(...):
    # HuggingFace式QKV permute attention
    return explicit_matmul_softmax_attention

def _sfdp_replacement_11(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：将显式转置后的链改为SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_11_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_11` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-12

```python
# torch/_inductor/fx_passes/fuse_attention.py:318
def _sfdp_pattern_12(...):
    # HuggingFace式permute并带dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_12(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：训练态保留dropout概率。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_12_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_12` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-13

```python
# torch/_inductor/fx_passes/fuse_attention.py:341
def _sfdp_pattern_13(...):
    # 三维bmm attention
    return explicit_matmul_softmax_attention

def _sfdp_replacement_13(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：用unsqueeze/squeeze映射到四维SDPA，且检查permute维。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_13_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_13` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-14

```python
# torch/_inductor/fx_passes/fuse_attention.py:358
def _sfdp_pattern_14(...):
    # BERT Large布局、加法mask
    return explicit_matmul_softmax_attention

def _sfdp_replacement_14(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：转置QKV并传递同dtype/bool mask。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_14_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_14` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-15

```python
# torch/_inductor/fx_passes/fuse_attention.py:384
def _sfdp_pattern_15(...):
    # DistilBERT masked_fill布尔mask
    return explicit_matmul_softmax_attention

def _sfdp_replacement_15(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：把0/1 mask规范为SDPA布尔mask。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_15_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_15` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。worker选择同编号注册输入：普通编号用half inference；3/4/6/7/9/12/28用half training和社区极低非零dropout=1e-11设计，保留requires_grad，仅测前向，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；新图必须另过精确目标与数值门禁，目前只是准备。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-104 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
