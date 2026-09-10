# T-112 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：1 个GPU-ready单元，0 个明确延期。

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

该pass利用sum/avg reduce-scatter的线性性质，把两个通信归并为一个。社区CUDA测试是真实compile并检查生成代码只有1个RS，但默认`world_size=1`，只能冻结结构改写，不能证明跨rank通信收益。NPU功能扩展固定2-rank HCCL：OFF必须2个RS、ON必须1个RS，全部rank数值一致。

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-112 \
  --gpu 2 \
  --wait-gpu
```

## 性能测例

社区没有目标性能benchmark。派生worker保持同一代数，扩为2-rank、每rank两个`[256,4096]` fp32输入，通过NCCL/HCCL执行；它是collective微图，不是FSDP模型端到端。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t112_dedup_reduce_scatter.py \
  --device npu \
  --phase functional
```

功能与生成代码人工签gate后才运行六臂性能；任一rank失败都会使整组无效。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| 无 | — | 唯一候选已进入GPU-ready合同。 |
