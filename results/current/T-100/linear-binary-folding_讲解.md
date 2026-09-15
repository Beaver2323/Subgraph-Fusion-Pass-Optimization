# T-100：冻结 Linear 的常量二元折叠

> 更新时间：2026-09-14 21:18 CST（UTC+08:00）。后端 `triton_experimental`，原社区功能与三轮性能已完成；未修改默认开关。

## 1. Pattern 做什么

冻结阶段已知参数和后续常量时，合法的 add/sub/mul/div 可以提前折进 Linear 的参数，减少运行期计算。
广播必须保持原输出形状；与 batch 相关的常量不能当作通道 bias 随意折叠。

```python
# PyTorch/torch/_inductor/fx_passes/binary_folding.py:479，folded_op；以下为公式示意
# add 的一种合法形式：
y = x @ weight.T + bias
out = y + constant                 # constant: [32]
# 冻结期计算 folded_bias = bias + constant
out = x @ weight.T + folded_bias
# mul/div会涉及权重和偏置缩放；具体合法性由原handler的extra_check决定。
```

```text
test_linear_binary_folding_cuda → torch.compile(no_grad)
 → freezing_passes → constant_fold → binary_folding_pass.folded_op
 → 再次constant_fold → NPU lowering/codegen → 设备执行 → 原数值/counter断言
```

## 2. 功能测例来自哪里，实际验证了什么

```python
# PyTorch/test/inductor/test_binary_folding.py:227，原社区方法
@inductor_config.patch({"enable_linear_binary_folding": True})
def test_linear_binary_folding(self):
    # 原完整参数乘积：2D/3D、add/sub/mul/div、scalar/tensor广播
    # 保留原atol=5e-5、rtol=5e-6和正例counter=1/负例counter=0。
    ...
```

GPU 原方法1/1、零skip；NPU先记录CUDA复制类未定义，再仅适配测试发现和设备。
NPU原方法1/1、零skip，共176次输出比较：160个正例折叠、16个非法广播负例不折叠。
64次2D正例与96次3D正例均有精确handler前后FX，16个负例保留原判据。

覆盖边界：原循环虽然枚举 `use_bias`，但没有把它传给 `nn.Linear`，实际始终默认 bias=True；
不能宣称覆盖 bias=False。原例为FP32冻结推理，没有FP16/BF16或参数梯度合同。
完整步骤见[复现报告](../../../issues/REF-linear-binary-folding-native/复现报告.md)与
[适配报告](../../../issues/REF-linear-binary-folding-native/适配报告.md)。

## 3. NPU OFF/ON 代码到底变了什么

性能图复用社区 Linear(3,32)、输入[4,3]、常量[32]，只测加法代表子图。
两臂仅 `enable_linear_binary_folding` 不同，均冻结、no_grad、fullgraph；独立进程和缓存。
ON的 `folded_op` 调用/图变化/计数均为1，OFF均为0。

```python
# issues/REF-linear-binary-folding-native/evidence/benchmark-20260914T211300/
# linear-binary-folding/off1/debug/torch_compile_debug/run_*/torchinductor/model*/output_code.py
extern_kernels.mm(arg3_1, _frozen_param3, out=buf0)
triton_unk_fused_add_addmm_0.run(buf1, _frozen_param1, _frozen_param2, 128, stream=raw_stream0)
# 同目录 on1/.../output_code.py：折叠后bias在冻结期已计算好
extern_kernels.addmm(_frozen_param4, arg3_1, _frozen_param3, alpha=1, beta=1, out=buf0)
```

OFF为一个NPU mm外部调用和一个Triton加法核；ON为一个NPU addmm外部调用。
这不是“生成了Triton GEMM融合核”，也不是CPU fallback。GPU/NPU均满足相同折叠合同，但具体kernel实现可不同。

## 4. 性能实测

未找到可直接控制此目标的社区benchmark，采用社区功能图派生子图端到端计时；不含编译、数据加载或完整模型。
OFF1/ON1/ON2/OFF2/OFF3/ON3六个新进程，每臂预热10、采样100，分位数从原样本独立重算。

| 时钟 | 指标 | OFF ms | ON ms | 改善率 |
|---|---|---:|---:|---:|
| 同步host | p50 | 0.43905 | 0.39357 | +10.36% |
| 同步host | p99 | 0.51024 | 0.49385 | +3.21% |
| NPU Event | p50 | 0.33065 | 0.28563 | +13.62% |
| NPU Event | p99 | 0.36259 | 0.31952 | +11.88% |

按既定三轮中位数口径为 `PERF_IMPROVED`；六臂内轮间p50波动比≤1.153。
功能支持无需产品修复。是否扩展默认开启仍需独立产品评审，不能用这个小图推断所有尺寸、dtype或模型均获益。

[功能复核](functional/linear-binary-folding.json)、[逐样本/显存/编译时间与性能汇总](performance_summary.json)、
[六臂FX/IR/生成代码清单](../../../issues/REF-linear-binary-folding-native/evidence/benchmark-20260914T211300/inventory.json)。
