# T-107 功能与性能测例讲解

> 更新时间：2026-09-15 01:40 CST（UTC+08:00）
> 状态：GPU 28/30精确目标确认，29映射待修正；NPU 28/30安装态原合同已实测但训练匹配未通过；原2个延期项保持。

29 的原用例带全零 mask，本轮实际成功 replacement 名称全部为 30，不计作 29 命中。
28/30的真实失败栈、FX/IR/codegen已归档到各自issue，继续定位训练分歧；29先重审合同，不通过改guard或关闭竞争pattern强行制造原生reference。
详见 [最新复核](../report/gpu_observer_review_20260914.md)。

29的非零mask专项已准备，复用冻结社区函数和输出/QKV梯度检查，不修改产品guard或竞争注册。
仅运行新增case的指令见[剩余任务与专项入口](../report/remaining_work_20260915.md)；这是新派生证据，尚待GPU执行，不覆盖原全零mask记录。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-107 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-28

```python
# torch/_inductor/fx_passes/fuse_attention.py:857
def _sfdp_pattern_28(...):
    # Visformer非连续QKV
    return explicit_matmul_softmax_attention

def _sfdp_replacement_28(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：融合前显式contiguous，避免布局错误。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_28_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_28` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-29

```python
# torch/_inductor/fx_passes/fuse_attention.py:876
def _sfdp_pattern_29(...):
    # BERT式分摊sqrt缩放并带mask
    return explicit_matmul_softmax_attention

def _sfdp_replacement_29(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：把Q/K双侧scale合并为scale平方。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_29_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_29` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-30

```python
# torch/_inductor/fx_passes/fuse_attention.py:906
def _sfdp_pattern_30(...):
    # pattern 29的无mask形式
    return explicit_matmul_softmax_attention

def _sfdp_replacement_30(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：融合双侧scale与_safe_softmax。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_30_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_30` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。worker选择同编号注册输入：普通编号用half inference；3/4/6/7/9/12/28用half training和社区极低非零dropout=1e-11设计，保留requires_grad，仅测前向，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；新图必须另过精确目标与数值门禁，目前只是准备。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-107 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-fuse-attention-sfdp-pattern-26` | `deferred-explicit-cuda-disabled-xpu-only` | 注册extra_check明确disable_cuda，GPU类也只在HAS_XPU_AND_TRITON时暴露该编号；禁止绕过CUDA关闭制造reference。 |
| `AU-fuse-attention-sfdp-pattern-27` | `deferred-explicit-cuda-disabled-xpu-only` | 注册extra_check明确disable_cuda，GPU类也只在HAS_XPU_AND_TRITON时暴露该编号；禁止绕过CUDA关闭制造reference。 |
