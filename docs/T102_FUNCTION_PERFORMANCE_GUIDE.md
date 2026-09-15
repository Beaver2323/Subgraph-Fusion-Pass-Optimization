# T-102 功能与性能测例讲解

> 更新时间：2026-09-15 20:20 CST（UTC+08:00）
> 状态：5/5 GPU精确编号确认，5/5 NPU原合同、安装态修复及性能处置完成；4改善、1回退。

最新逐pattern代码、数值范围、GPU/NPU对照、OFF/ON生成代码与实测见[T-102交付索引](../results/current/T-102/README.md)。
逐例旧失败栈与FX/IR/output_code位于 `issues/REF-sfdp-pattern-编号-native/安装态基线复核.md`，不覆盖；
共同注册修复及五原方法、边界证据见[修复验证](../issues/REF-sfdp-pattern-1-native/修复验证报告.md)。
3/4原社区没有随机输出/梯度数值比较，5号实际为数学展开，三项部分对齐限制仍保留。

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

意图：无mask、无dropout，scale=1/inv_scale。GPU原方法与精确编号已经验证。旧NPU训练图与预生成注册不匹配；现按活动NPU decomposition重建训练匹配图，安装态原方法20次Tensor比较、8次精确改写通过，独立性能PERF_IMPROVED。具体代码和调用链见[根因](../issues/REF-sfdp-pattern-1-native/根因分析.md)与[修复验证](../issues/REF-sfdp-pattern-1-native/修复验证报告.md)。

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

意图：无mask、无dropout，直接保留scale_factor。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_2_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_2` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：把训练态dropout概率交给SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_3_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_3` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：把显式matmul/softmax/dropout链收为SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_4_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_4` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：mask转为Q dtype后进入SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_5_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_5` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。T-102五项均使用half training注册输入、保留requires_grad但仅计前向；1/2/5测本次修复的训练路径，3/4采用社区极低非零dropout=1e-11设计，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；本批五项已各过精确目标、数值门禁及六臂计时，4改善、5号回退。compile_ms在joint lazy_init之后开始，不含完整冷启动。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

真实预检问题及最小适配见[1号推理OFF被邻接接替](../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)与[2号scale占位转Python标量](../issues/REF-sfdp-pattern-2-native/性能输入适配报告.md)；不删除相邻注册，不放宽产品guard。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-102 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
