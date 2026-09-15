# T-106 功能与性能测例讲解

> 更新时间：2026-09-14 23:59 CST（UTC+08:00）
> 状态：4/4 GPU精确编号确认；21/23/24号NPU原例通过（Tensor比较分别2/6/1次）；22号已改图但实际数值失败。23/24六臂计时和归档已完成，分别PERF_IMPROVED/PERF_REGRESSED，本批2/4正式闭环。

[23号功能性能讲解](../results/current/T-106/pattern-23_讲解.md)、[24号数学路径与回退讲解](../results/current/T-106/pattern-24_讲解.md)含代码框、真实生成代码、原样本和GPU/NPU差异。

21号性能OFF被22号接替，不能把整轮attention关闭后所得时延算作21号收益，见[归因阻断](../issues/REF-sfdp-pattern-21-native/性能归因阻断.md)。
22号已定位共享mask的extract_slice广播尺寸错误；仅改生成副本的4处尺寸后，3种子24次逐kernel比较及最终输出均零误差。
该诊断不等于通用修复完成：带存储边界保护的codegen隔离候选原例正在运行，仍待完整结果和邻接验证，未部署安装态。

21/22/24各自的原件和`代码断言适配分析.md`已放对应issue；代码符号断言适配未改变产品选择或数值容差。
21/24复验通过；22第一次输出比较3772/4096元素超差，现已另立[数值失败分析与调用栈](../issues/REF-sfdp-pattern-22-native/数值失败分析.md)。
不能把最初的符号断言问题与随后实际暴露的数值问题混为同一个结论。原1个延期项保留。

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

意图：融合为无缩放SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_21_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_21` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：保持多输出合同。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_22_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_22` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：用attn_mask=None的SDPA。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_23_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_23` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

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

意图：恢复四维后以scale=1融合。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_24_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_24` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。worker选择同编号注册输入：普通编号用half inference；3/4/6/7/9/12/28用half training和社区极低非零dropout=1e-11设计，保留requires_grad，仅测前向，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；新图必须另过精确目标与数值门禁，目前只是准备。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-106 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-fuse-attention-sfdp-pattern-25` | `deferred-explicit-cuda-disabled-xpu-only` | 注册extra_check明确disable_cuda，GPU类也只在HAS_XPU_AND_TRITON时暴露该编号；禁止绕过CUDA关闭制造reference。 |
