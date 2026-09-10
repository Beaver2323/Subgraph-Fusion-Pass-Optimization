# T-100 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：1 个GPU-ready单元，0 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-100 \
  --gpu 2 \
  --wait-gpu
```

先运行冻结 PyTorch revision 的原生社区测例；只有真实设备/backend/采集阻断才进入最小适配审核。

## AU-binary-folding-folded-op

```python
# torch/_inductor/fx_passes/binary_folding.py:479
def folded_op(match, *args, **kwargs):
    counters["inductor"]["binary_folding"] += 1
    # 把Linear后的add/sub/mul/div常量预折入weight/bias
```

旧索引漏掉了CUDA复制类。原生功能测例覆盖2D/3D Linear、四种binary、scalar/tensor广播正例和bad-shape负例，并逐项断言counter。Conv版本受CUDNN accuracy skip，不用于本批分母。性能测例固定Linear(3,32)+[32]常量add，freezing始终开启，仅切换linear binary folding。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 本批唯一候选已纳入GPU-ready合同。 |

## NPU 功能与性能入口

GPU分母人工复核后，从NPU服务器的固定tmp目录先跑功能预检：

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t091_t100_performance.py \
  --task T-100 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，再使用 `--phase benchmark --gate-root <目录>` 执行固定六臂。
