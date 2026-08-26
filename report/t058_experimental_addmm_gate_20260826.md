# T-058 experimental addmm gate 代表 cohort（2026-08-26）

## 结论

`triton_experimental` 默认关闭的 add+mm→addmm fusion 在首个
FP16/static/contiguous/vector-bias cohort 上可用且明显有益：融合侧真实命中 pattern，并把
`extern_kernels.mm + Triton add` 两步改成单个 `extern_kernels.addmm`。三轮 fresh-process 交错
采样的 p50 中位数改善 `22.17%`，p99 中位数改善 `14.05%`，峰值 allocated 不变。

首个 verdict 为 `representative-supported-beneficial-gate-shrink-coverage-pending`；随后完成的
11 项覆盖全部通过，第二个 unaligned cohort 的 p50 中位数也改善 `17.49%`。P-018 最小 source
gate 已实施并通过无设备探针，状态为 `source-verified-wheel-pending-host-tail-monitor`；installed
wheel 仍未改变。现有 NPU addmm lowering
已承接成功，不需要手写 Triton addmm。

## 测试边界

- 环境：`/home/z50063656/Benchmark/env.sh`，installed P-013 wheel；
- backend：fresh-process `triton_experimental`，物理 NPU 1，cwd `/home/z50063656/tmp`；
- 输入：`M=192,K=256,N=320`，FP16 contiguous，bias shape `(320,)`；
- runner：`t058_experimental_addmm_gate_probe.py`，当前 SHA256 `c2d13e6d...1d49b`；
- 两侧都显式 `elide_int_float_int=False`，独立 compile/Triton cache，强制关闭 Inductor cache；
- 性能：三轮交错 `D1,E1,E2,D2,D3,E3`，每个进程 warmup 10/runs 100；D 为默认 disabled，E 为
  audit-only enabled。

enabled 不能只在 `torch_npu` 导入后把 config 改成 False：backend 激活已经不可逆地把 entry 的
`extra_check` 换成 False lambda。有效审计路径先以 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 导入 torch，
保存两个唯一上游 addmm entry 的原始 check，再显式导入 torch_npu 并恢复这两个 check。进程退出即
回收，不修改产品源码。这既提供了可控 A/B，也再次印证 T-057 的 backend 状态隔离缺口。

## 功能与生成代码

共同输入上两侧相对 CPU/NPU eager 都通过 FP16 `rtol=atol=0.01`：

| 模式 | pattern delta | 生成结构 | 对 NPU eager 最大绝对差 | 对 CPU 最大绝对差 |
|---|---:|---|---:|---:|
| disabled | 0 | `1×extern mm + 1×Triton add` | `0` | `0.03125` |
| enabled | 1 pattern / 2 nodes | `1×extern addmm` | `0.0625` | `0.0625` |

enabled 的误差不是 0，说明融合 addmm 与分离 mm+add 的 FP16 舍入路径不同；当前仍在合同容差内，
但覆盖扩展必须继续记录，不能只报性能。enabled 图没有 experimental Triton kernel，所以
generated wrapper 不需要导入 experimental heuristic marker；backend 的作用体现在 gate/registry
和 NPU extern lowering，不能把“无 marker”误判为回退。

有效功能结果：

| 模式 | result JSON SHA256 | generated code SHA256 |
|---|---|---|
| disabled retry2 | `a12b6a5d...9772` | `69ea6301...5417` |
| enabled retry1 | `09cd0537...e574` | `127881f3...ae0a` |

按图模式诊断要求，两侧 `TORCH_COMPILE_DEBUG` 的 transformed FX、IR 和 output_code 均已检查。
disabled output_code 有 `extern_kernels.mm` 和 `triton_unk_fused_add_0`；enabled output_code 只有
`extern_kernels.addmm`，不是 CPU fallback。

## 三轮性能

| round | disabled p50 ms | enabled p50 ms | p50 改善 | disabled p99 ms | enabled p99 ms | p99 改善 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.260650` | `0.205670` | `21.09%` | `0.331550` | `0.263050` | `20.66%` |
| 2 | `0.265520` | `0.201800` | `24.00%` | `0.306050` | `0.291360` | `4.80%` |
| 3 | `0.253235` | `0.202860` | `19.89%` | `0.277450` | `0.225180` | `18.84%` |
| 中位数 | `0.260650` | `0.202860` | `22.17%` | `0.306050` | `0.263050` | `14.05%` |

两侧三轮 `max_allocated_bytes` 都为 `510,976`，steady allocated 都为 `387,584`。首次编译+运行
中位数从 disabled 的 `19,628.63 ms` 降到 enabled 的 `925.31 ms`，改善约 `95.29%`；主要原因是
融合侧不再编译独立 Triton add kernel。该数值在强制禁缓存的 audit launcher 环境下成立，不外推
为正式生产首编 SLA。

结果 JSON SHA256：

| D1 | E1 | E2 | D2 | D3 | E3 |
|---|---|---|---|---|---|
| `ee02804b...ff5` | `e0e75633...82d` | `f81890d2...2bd` | `136e1180...e55` | `8022da96...2b8` | `22a78370...995` |

## 中性与失败尝试

- 初版 disabled 图已经编译完成，但 runner 用扁平 argmax 索引二维 tensor，取到一整行并在
  `.item()` 失败；修正为 flatten 后以 retry 新目录复验。它是 audit runner 失败，不是产品失败。
- 第一个 enabled 尝试在 torch_npu 已激活后才改 config，结果仍是 mm+Triton add。它证明 config
  是“激活时生效”，不是 live reversible 开关；不纳入性能 enabled 数据。

## capability 覆盖扩展

`t058_experimental_addmm_coverage_probe.py` 当前 SHA256 为 `4ba58c02...0882d`。它在每个 fresh
process 中确认 `_InductorNpuRegistry._loaded_backend=triton_experimental`，并用 generated code
同时判定正例必须单 addmm、负例必须保持 mm+add。新增 11/11 结果如下：

| case | dtype/shape/layout | 模式 | 预期/观察 | 数值结果 |
|---|---|---|---|---|
| row bias | FP16 shape-A contiguous | inference | fused/fused | 通过，max vs eager `0.0625` |
| full bias | FP16 shape-A contiguous | inference | fused/fused | 通过，max vs eager `0` |
| vector | BF16 shape-A contiguous | inference | fused/fused | 通过，max vs eager `0.5` |
| vector | FP32 shape-A contiguous | inference | fused/fused | 通过，max vs eager `1.335144e-5` |
| vector | FP16 unaligned | inference | fused/fused | 通过，max vs eager `0.03125` |
| vector | FP16 real-transposed | inference | fused/fused | 通过，stride `(1,M)/(1,K)` |
| vector | FP16 shape-A→`+8` | dynamic replay | fused/fused | 两 shape 通过 |
| vector | FP16 small | forward/backward | fused/fused | 输出通过，3 个 grad max diff 均 0 |
| full bias | FP16 small | forward/backward | fused/fused | 输出/3 个 grad max diff 均 0 |
| scalar add | FP16 small | negative | unfused/unfused | 通过 |
| mixed FP16 mm + FP32 bias | small | negative | unfused/unfused | 通过 |

BF16 的 `0.5` 仍在 `rtol=atol=0.03` 内，但它说明最终回归不能只覆盖 FP16。每份 debug 的 forward
output_code 均已检查；training 还保存独立 backward FX/IR/output_code。

## unaligned 第二性能 cohort

FP16 `M=191,K=255,N=319` 继续三轮 `D1,E1,E2,D2,D3,E3`：

| round | disabled p50 ms | enabled p50 ms | disabled p99 ms | enabled p99 ms |
|---:|---:|---:|---:|---:|
| 1 | `0.246800` | `0.211995` | `0.409460` | `3.377270` |
| 2 | `0.244365` | `0.203635` | `0.275360` | `0.239110` |
| 3 | `0.302775` | `0.203185` | `0.399980` | `0.255260` |
| 中位数 | `0.246800` | `0.203635` | `0.399980` | `0.255260` |

p50/p99 中位数改善 `17.49%/36.18%`，两侧 max allocated 同为 `506,368 B`。但 E1 有一次
`3.377 ms` host p99，不能删除。额外 D4/E4 使用 NPU Event：device p50
`0.222670→0.179520 ms`（改善 `19.38%`），device p99 `0.244940→0.197140 ms`（改善
`19.51%`）；对应 host p50 改善 `16.58%`，host p99 则回退 `9.39%`。这表明融合 kernel/device
时间稳定有益，host 同步仍可能出现长尾；最终 installed 回归必须继续监控 host p99。

## 下一步与修改边界

上述功能层和第二性能 cohort 已完成。P-018 的最小 source 方案已实施：把
`disable_addmm_fusion` source 默认改为 False，让已有上游 validity check 与 NPU extern addmm
lowering工作；显式 opt-out 路径改成保存 original check 的幂等/可恢复 wrapper。不得修改 PyTorch
通用 pattern、addmm lowering 或 Triton kernel。

device-independent `t058_p018_source_gate_probe.py`（SHA256 `ae0bcbf6...5d7bf`）已从 current
source AST 验证两个唯一 entry：默认不改、显式 disable、重复调用不叠加、切回 False 恢复 original、
重复恢复均通过。修改后 config.py/fx_passes.py SHA256 分别为 `1e9e62ec...c06b33`、
`e5bc23d4...ec95e`，语法与 diff whitespace 检查通过。

shared tree 可隔离前不构建 wheel；installed P-013 仍执行旧默认关闭。最终 installed wheel 必须
重跑 11 项功能矩阵、两性能 cohort、显式 opt-out 和 host p99，才能把 wheel-pending 关闭。
