# T-103 功能与性能测例讲解

> 更新时间：2026-09-14 22:50 CST（UTC+08:00）
> 状态：5/5 GPU精确编号确认；6～10号NPU安装态原合同均已实测，训练匹配断言未通过，尚无性能门禁。

逐例原断言、失败栈、目标观察和生成代码见 `issues/REF-sfdp-pattern-编号-native/安装态基线复核.md`。
8/10号部分推理分支命中的是邻接编号，不能把总counter当作本编号的完整NPU覆盖。

NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-103 \
  --gpu 2 \
  --wait-gpu
```

社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。

## 功能测例

### AU-fuse-attention-sfdp-pattern-6

```python
# torch/_inductor/fx_passes/fuse_attention.py:150
def _sfdp_pattern_6(...):
    # 带mask和dropout的除法缩放
    return explicit_matmul_softmax_attention

def _sfdp_replacement_6(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：同时保留mask、scale与dropout语义。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_6_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_6` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-7

```python
# torch/_inductor/fx_passes/fuse_attention.py:171
def _sfdp_pattern_7(...):
    # QKV先permute且softmax上采样FP32
    return explicit_matmul_softmax_attention

def _sfdp_replacement_7(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：避免SDPA内部额外布局复制并保留dropout。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_7_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_7` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-8

```python
# torch/_inductor/fx_passes/fuse_attention.py:208
def _sfdp_pattern_8(...):
    # pattern 7的无dropout形式
    return explicit_matmul_softmax_attention

def _sfdp_replacement_8(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：保留permute和FP32 softmax意图。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_8_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_8` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-9

```python
# torch/_inductor/fx_passes/fuse_attention.py:237
def _sfdp_pattern_9(...):
    # 先缩放Q、再matmul并带dropout
    return explicit_matmul_softmax_attention

def _sfdp_replacement_9(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：识别缩放位置不同但等价的attention链。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_9_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_9` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

### AU-fuse-attention-sfdp-pattern-10

```python
# torch/_inductor/fx_passes/fuse_attention.py:267
def _sfdp_pattern_10(...):
    # pattern 9的无dropout形式
    return explicit_matmul_softmax_attention

def _sfdp_replacement_10(...):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径
```

意图：融合Q预缩放与FP32 softmax。GPU执行 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_10_gpu`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_10` counter/FX对应起来。按本文件顶部实际执行状态区分已测与待办；验收仍须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。

## 性能测例

社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。worker选择同编号注册输入：普通编号用half inference；3/4/6/7/9/12/28用half training和社区极低非零dropout=1e-11设计，保留requires_grad，仅测前向，避免置零后冒用邻接编号。它属于tracker派生微图，不是社区原生benchmark、完整训练或模型端到端；新图必须另过精确目标与数值门禁，目前只是准备。详见[阶段选择与社区依据](../report/attention_performance_contract_review_20260914.md)。

2026-09-14 20:48 CST补强：注册用torch.empty不得直接参与数值/计时；固定seed、正态std=0.25初始化张量，保留shape/stride/dtype及0D标量常量。
两臂都启用joint passes，OFF只删除精确编号entry；不得关闭整轮joint。如果相邻attention接替OFF，则拒绝收益归因。
精确目标由只读entry观察器确认，不依赖未开启debug时为空的per-pattern counter。16/17/29的原例目标映射未闭环，worker拒绝执行。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \
  --task T-103 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF只删除目标编号注册，其余joint优化保持；结论只归属于已审查的派生微图。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批全部候选已进入GPU-ready合同。 |
