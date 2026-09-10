# T-097 功能与性能测例讲解

> 更新时间：2026-09-10 21:38:54 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，4 个候选明确延期。

NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。

## 功能测例

本批经审核没有可冻结的原生GPU直接合同。可做零设备静态校验，但一键入口会拒绝实际空跑：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-097 \
  --validate-only
```

这不是SKIP或PASS；所有旧ID均保留为非计数延期记录。

## 性能测例

没有通过GPU功能门禁的合同，因此不派生性能测例、不制造ON路径，性能状态为不适用。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-replace-random-fuse-offset-creation` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-replace-random-fuse-seed-creation` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-replace-random-replace-randint` | `deferred-no-direct-community-test` | 冻结revision未找到直接社区测例。 |
| `AU-replace-random-replace-random` | `deferred-fake-cpu-structural-test` | 现有测例使用make_fx tracing_mode=fake和CPU样例，只验证stack metadata，不是设备执行或随机语义合同。 |
