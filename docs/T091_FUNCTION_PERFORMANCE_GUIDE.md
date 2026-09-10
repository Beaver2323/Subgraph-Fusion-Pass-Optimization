# T-091 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：1 个GPU-ready单元，4 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-091 \
  --gpu 2 \
  --wait-gpu
```

先运行冻结 PyTorch revision 的原生社区测例；只有真实设备/backend/采集阻断才进入最小适配审核。

## AU-split-cat-normalize-stack-default

```python
# torch/_inductor/fx_passes/split_cat.py:383
dim = get_arg_value(node, 1, "dim") or 0
...
new_node = graph.call_function(node.target, args=(tensors,), kwargs={"dim": dim})
counters[backend]["normalization_pass"] += 1
```

功能测例用两个 `[4,4]` GPU 张量执行 `torch.stack(axis=1)`；意图是把 numpy 兼容关键字和负维统一成规范 FX 形式。GPU/NPU都需证明数值一致和实际改写。性能测例复用相同图，唯一变量是 `normalization_pass` OFF/ON；它是微图，不是模型端到端。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-split-cat-normalize-split-default` | `deferred-cpu-only-community-test` | 两个映射测例都在CPU张量上运行，不能冻结原生GPU合同。 |
| `AU-split-cat-normalize-split-default-aten` | `deferred-cpu-only-community-test` | Aten normalization测例使用CPU输入，且一个总计数不能独立归属三个split handler。 |
| `AU-split-cat-normalize-split-with-size-default-aten` | `deferred-cpu-only-community-test` | 共享CPU测例没有逐handler断言，不能重复计为独立GPU单元。 |
| `AU-split-cat-normalize-squeeze-default` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |

## NPU 功能与性能入口

GPU分母人工复核后，从NPU服务器的固定tmp目录先跑功能预检：

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t091_t100_performance.py \
  --task T-091 \
  --phase functional \
  --device npu
```

功能原件人工签gate后，再使用 `--phase benchmark --gate-root <目录>` 执行固定六臂。
