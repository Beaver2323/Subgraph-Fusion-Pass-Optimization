# P0 单 Pass A/B：首个 fp16 Shape（2026-08-20）

## 结论

在物理 NPU 2 空闲、同 backend、fresh process、Inductor cache 关闭的条件下，两个已确认触发的 pass 都在三轮独立采样中表现出稳定正收益：

| pass / backend | current 图 | disabled baseline | current p50 中位数 | baseline p50 中位数 | p50 延迟下降 | p99 延迟下降 |
|---|---|---|---:|---:|---:|---:|
| addmm fusion / `default` | 1×`aten.addmm` | `mm + add` | 0.219265 ms | 0.269925 ms | 18.8% | 16.7% |
| mm_plus_mm / `triton_experimental` | 1×`_mm_plus_mm` extern | 2×`mm` + `add` | 0.238375 ms | 0.291095 ms | 18.1% | 18.4% |

这是 `fp16`、固定静态 shape 的首个性能点，不代表所有 dtype、shape、layout 或动态 shape。矩阵 performance status 已更新为 `beneficial-*-fp16-shape-A`，最终 verdict 保持 `not-run`，等待覆盖扩展。

## 方法

- addmm：`(192,256) @ (256,320) + bias(320,)`。
- mm_plus_mm：两组 `(192,256) @ (256,320)` 后相加。
- current 和 disabled 各自使用 fresh worker；disabled 仅在该 worker 内把目标 pattern 的 `extra_check` 置 False，进程退出即恢复。
- 每个模式每轮 warmup 10、runs 100，逐次 NPU synchronize；共 3 轮。
- 第 2 轮反转执行顺序，以降低固定顺序偏差。
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`，保证每个 worker 真实重新编译并运行 pass。
- 物理 NPU 0/1 在采样前出现外部任务，因此改用无运行进程的物理 NPU 2；采样后复核 NPU 2 仍无运行进程。

## 三轮结果

### addmm fusion / default

| 轮次 | current p50 | disabled p50 | current p99 | disabled p99 |
|---|---:|---:|---:|---:|
| 1 | 0.219265 ms | 0.284825 ms | 0.237410 ms | 0.298900 ms |
| 2 | 0.217130 ms | 0.269925 ms | 0.237830 ms | 0.285640 ms |
| 3 | 0.229520 ms | 0.264035 ms | 0.251520 ms | 0.279810 ms |

三轮 current mean 的中位数为 0.221421 ms，disabled 为 0.270909 ms。current 首次编译+执行中位数为 13.742 s，disabled 为 20.316 s。这里的峰值 allocator 记录为 1,249,792 对 201,961,472 bytes，但 reset 在编译前，disabled 又触发 Triton autotune，因此该值是“编译+首跑峰值”，不能解释成纯稳态 runtime 内存。

disabled 第 3 轮首次尝试出现一次 `NoTritonConfigsError`，根异常是 Triton 预编译 worker 的 `OSError: could not get source code`；唯一一次受控重试成功。失败结果保留在 `addmm_r3_disabled/p0_gate_probe.json`，重试结果在 `addmm_r3_disabled_retry1/`。

### mm_plus_mm / triton_experimental

| 轮次 | current p50 | disabled p50 | current p99 | disabled p99 |
|---|---:|---:|---:|---:|
| 1 | 0.254355 ms | 0.290240 ms | 0.281820 ms | 0.304120 ms |
| 2 | 0.236265 ms | 0.291095 ms | 0.254010 ms | 0.329320 ms |
| 3 | 0.238375 ms | 0.295635 ms | 0.259290 ms | 0.317660 ms |

三轮 current mean 的中位数为 0.241171 ms，disabled 为 0.293644 ms。current 首次编译+执行中位数为 14.879 s，disabled 为 19.911 s。编译+首跑峰值稳定为 1,511,936 对 2,253,824 bytes。

## 证据位置

- JSON：`results/p0_ab_20260820/*/p0_gate_probe.json`
- 第 1 轮完整 debug graph：`addmm_default_{current,disabled}/debug/`、`mmplus_experimental_{current,disabled}/debug/`
- A/B 测试侧开关：`run_p0_gate_probe.py --disable-target-pass`

## 下一步

1. 扩展 fp16/bf16/fp32、对齐/非对齐、较小/较大矩阵和非连续输入。
2. addmm 补 `(M,N)`、`(1,N)` bias、alpha/beta 和 dtype mismatch 负例。
3. mm_plus_mm 补两组不同 K、动态 shape、转置/非连续输入。
4. 将 Triton worker 的间歇性 source-inspection 失败单独纳入工具链稳定性记录，不用重试后的成功掩盖失败。
