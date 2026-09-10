# T-087 功能与性能测例讲解

> 更新时间：2026-09-10 23:14:54 CST（UTC+08:00）
> 状态：2 个单元已准备，等待原生 GPU reference；另 2 个候选延期。

NPU 的功能、修复验证和性能统一使用 `triton_experimental`，必须在导入 `torch`/`torch_npu`
前选择后端，OFF/ON 每臂使用新进程。GPU 先运行冻结 revision
`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` 的原生社区测例。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-087 \
  --gpu 2 \
  --wait-gpu
```

回传 `t087-reference-results/latest-text-handoff.json`；若脚本提示 split，则回传
`latest/text-handoff-parts/manifest.json` 和全部 `part-*.json`。

## AU-post-grad-reorder-for-locality

功能测例来自 `test/inductor/test_reorder_for_locality_in_training.py`。实际入口先检查总开关，再检查
训练子开关：

```python
# torch/_inductor/fx_passes/post_grad.py:193
if config.reorder_for_locality and (
    is_inference or config.reorder_for_locality_in_training
):
    GraphTransformObserver(gm, "reorder_for_locality").apply_graph_pass(
        reorder_for_locality
    )
```

意图是把生产者移动到唯一消费者附近，改善融合和缓存局部性，同时保护 RNG、mutation 和 collective
顺序。社区 `_TwoBranch` 使用两组 `[16,16]` 参数和 `[8,16]` CUDA 输入，执行真实前向、反向和 SGD
step：ON 必须实际改变节点顺序，OFF 不调用；梯度和更新后参数必须一致。GPU 与 NPU 对比看同一图的
调用/改写与数值，不把“调用了”当作“生效”。

性能测例复用该模型、shape 和 seed，测一次 compiled forward+backward；唯一变量是
`reorder_for_locality_in_training=false/true`，总开关固定 true。它是目标训练子图，不是完整训练端到端。

## AU-post-grad-respecialize-current-device

```python
# torch/_inductor/fx_passes/post_grad.py:145
nodes = graph.find_nodes(
    op="call_function", target=torch.ops.coor.current_device.default
)
device = _coor_current_device()
for node in nodes:
    # 用运行时当前设备替换device-valued FX node，随后删除该node
    ...
```

该 pass 使 compile-on-one-rank 图保持跨 rank 可复用，又在 GraphLowering 前消除 Inductor 不支持的
device-valued node。原生功能测例在真实 CUDA 上编译 `[2,8]` 输入，检查 generated code 使用
`torch.cuda.current_device()` 且没有 `cuda:N`/固化 index。NPU 最小适配只能替换设备断言并保持
同样合同。

它没有合法性能 OFF：移除 pass 会让不可 lowering 的 device node 继续下沉，得到的是预期编译失败，
不是等价基线。因此性能状态为 `PERF_EXEMPT`，只保存 FX、generated code 与正确性证据。

## 延期候选

| 候选 | 当前证据 | 延期原因 |
| --- | --- | --- |
| `replace_collectives_with_low_contention` | CPU FX + mock gate | 无真实 GPU collective/编译合同 |
| `stable_sort` | 直接调用纯 FX helper | 未证明真实 post-grad GPU 入口 |

## NPU 功能与性能入口

GPU 分母通过后运行功能预检：reorder执行OFF/ON两臂；无合法OFF的
respecialize-current-device只执行ON功能臂，仍会保存真机数值、FX和generated-code证据：

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t087_t090_performance.py \
  --task T-087 \
  --phase functional \
  --device npu
```

功能原件人工复核并签 gate 后，改为 `--phase benchmark --gate-root <目录>` 执行固定六臂。
