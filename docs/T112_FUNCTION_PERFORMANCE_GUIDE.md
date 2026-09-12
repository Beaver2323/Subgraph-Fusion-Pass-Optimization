# T-112 功能与性能测例讲解

> 更新时间：2026-09-11 12:32 CST（UTC+08:00）
> 状态：GPU 原生 1/1 通过；与 T-084 同合同，保留 ID/证据，独立贡献 0。

NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。

## 功能测例

### AU-fsdp-get-dedup-rs

```python
# torch/_inductor/fx_passes/fsdp.py:101,167
# before: wait(RS(a)) + wait(RS(b))
combined = aten.add.Tensor(input_a, input_b)
rs = c10d.reduce_scatter_tensor.default(combined, reduce_op, group_size, group_name)
return c10d.wait_tensor.default(rs)
```

该pass利用sum/avg reduce-scatter的线性性质，把两个通信归并为一个。社区CUDA测试是真实compile并检查生成代码只有1个RS，但默认`world_size=1`，只能证明结构改写，不能证明跨rank通信收益。

该内部 builder 由 `post_grad.py -> fsdp.dedup_fsdp_reduce_scatter -> _get_dedup_rs_pass` 调用，
与 T-084 的开关、原生方法、2→1 RS 数学合同完全一致。因此当前 canonical 为
`AU-post-grad-dedup-reduce-scatters`，不再新增能力分母或重复跑 NPU/性能。已有 GPU 包保留于
`results/incoming/T-112/text-handoff.json`，本轮未重新认证 T-084 历史 NPU/性能。

以下 GPU 命令仅用于同合同重跑，不是当前必做队列：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-112 \
  --gpu 2 \
  --wait-gpu
```

## 性能测例

原准备件中的派生 worker 保持同一代数，扩为2-rank、每rank两个`[256,4096]` fp32输入；
它是 collective 微图，不是 FSDP 模型端到端。准备记录保留，但 `run_t112_dedup_reduce_scatter.py`
现只允许 `--validate-only`，实际 NPU/性能执行会拒绝并指向 T-084。需要新增测量时应归属 canonical 合同，
核对版本、后端、输入、门禁生命周期和测量方法，再运行真实多 rank；不能把单 rank 通过当通信收益。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `_get_dedup_rs_pass` | duplicate-contract-evidence-retained | 唯一候选是 T-084 同一优化的内部入口，不重复计数。 |
