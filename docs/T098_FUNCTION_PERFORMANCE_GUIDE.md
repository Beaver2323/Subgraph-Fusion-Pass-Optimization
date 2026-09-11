# T-098 功能与性能测例讲解

> 更新时间：2026-09-11 11:14 CST（UTC+08:00）
> 状态：1 个GPU-ready单元，2 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-098 \
  --gpu 2 \
  --wait-gpu
```

先运行冻结 PyTorch revision 的原生社区测例；只有真实设备/backend/采集阻断才进入最小适配审核。

该社区方法包含112组参数，TF32双臂可达到224组，因此生成的FX/IR/代码较多。若已经运行完成，
旧handoff出现上百片，pull更新后执行以下命令重新压缩回传，不重新运行GPU：

```bash
cd /data/z50063656/tmp
/data/z50063656/envs/PassGPURef/bin/python \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/reexport_reference_text.py \
  --task T-098
```

复制输出 `handoff_upload_input` 指向的 manifest 及同目录全部分片。原run和旧包不覆盖，
1.4格式仍可恢复完整评审范围FX/IR/output_code原文，详见[传输指南](GPU_TEXT_HANDOFF.md)。

## AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-inlined

```python
# torch/_inductor/fx_passes/efficient_conv_bn_eval.py:24
weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
coefff_on_the_fly = bn.weight.view_as(weight_coeff) * weight_coeff
weight_on_the_fly = conv.weight * coefff_on_the_fly
bias_on_the_fly = bn.bias + coefff_on_the_fly.flatten() * (conv_bias - bn.running_mean)
return functional_call(conv, {"weight": weight_on_the_fly, "bias": bias_on_the_fly}, x)
```

功能测例是社区真实CUDA完整乘积，覆盖Linear/Conv/ConvTranspose、bias、SyncBN、单/多用户、inlined/decomposed，并检查前向、梯度、SGD后结果和精确counter。两个handler合为一个行为合同。性能测例从乘积中固定Conv2d+BN2d代表图，测forward+backward；不是完整模型端到端。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform` | `deferred-no-direct-call-module-attribution` | 社区test_basic在Dynamo/ATen图上覆盖inlined/decomposed路径，未独立证明CallModule handler。 |
| `AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-decomposed` | `merged-into-inlined-contract` | 同一个test_basic_cuda循环以decompose_nn_module false/true覆盖两条路径，合并到一个Conv-BN合同，禁止重复执行和重复计分母。 |

## NPU 功能与性能入口

GPU分母人工复核后，从NPU服务器的固定tmp目录先跑功能预检：

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t091_t100_performance.py \
  --task T-098 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，再使用 `--phase benchmark --gate-root <目录>` 执行固定六臂。
