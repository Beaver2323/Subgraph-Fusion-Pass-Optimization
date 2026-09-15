# T-105 功能与性能测例讲解

> 更新时间：2026-09-15 01:37 CST（UTC+08:00）
> 最新口径：5个跟踪ID、4个独立合同；17是15的推理去重别名，不再要求独立命中。18/19已完成性能处置，16待针对性GPU补测，20仍有OFF数值阻断。下列旧阶段观察保留，不作为当前总数。
> 状态：GPU 18～20精确目标确认，16/17映射待修正；NPU 18/19/20原例已通过。18已完成独立OFF/ON及六臂性能：PERF_IMPROVED；19混合dtype ON编译失败，20额外FP16微图OFF数值失败，均未计时。

本批目前2/4独立合同正式闭环。[18号源码、GPU/NPU行为与性能讲解](../results/current/T-105/pattern-18_讲解.md)。

16的专项运行指令、输入最小改动、精度检查范围，以及17的源码去重证明见[剩余任务与专项入口](../report/remaining_work_20260915.md)。专项是新证据，不回填原例；不要重复原来只命中14/5的入口来证明16。

新增执行缺口：[19号FP16/FP32 mask域](../issues/REF-sfdp-pattern-19-native/性能精度域缺口.md)、
[20号OFF数值异常](../issues/REF-sfdp-pattern-20-native/性能OFF数值失败.md)。原例没有的数值检查不能因结构通过回填成PASS。

18号原例有容器输出数值比较，旧观察器漏记Tensor叶子；原合同复验已记录12个叶子比较通过；
19/20号原例确实没有输出/梯度比较，不能记精度PASS。详见[18号观察器修正](../issues/REF-sfdp-pattern-18-native/数值观察器修正说明.md)
及各issue的`安装态基线复核.md`。这两类“零比较”原因不同。

本轮真实观察：16 的入口只命中 14/5，17 只命中 15；下面的 pattern 源码意图不等于本轮已命中。
不能把旧计划里对 16 的预期写成设备事实，也不应重新盲跑这两个相同入口。
逐项证据及不阻塞同批其他单元的下一步见 [最新复核](../report/gpu_observer_review_20260914.md)。

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

意图：CUDA命中后故意保留数学路径保证数值；非CUDA可转SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_16_inference_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_16` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：规范布尔mask并保留dropout。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_17_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_17` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：融合attention同时保持多输出K/V。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_18_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_18` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：先合成mask再交给SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_19_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_19` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：覆盖不同缩放位置。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_20_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_20` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。worker选择同编号注册输入：普通编号用half inference；3/4/6/7/9/12/28用half training和社区极低非零dropout=1e-11设计，保留requires_grad，仅测前向，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；新图必须另过精确目标与数值门禁，目前只是准备。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-105 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
