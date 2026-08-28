# T-065 triton_experimental header tiling/odometer 验证报告

## 1. 结论

T-065 / TE-CG-008 已完成。五项 header codegen 配置均完成静态可达性和 fresh-process
NPU 开/关覆盖；最终正确性矩阵 10/10、三个有效开关的三轮 paired 性能 18/18 通过。

- `reduce_xblock_by_input_stride=True` 保留：代表 reduction 的 device P50/P99 中位改善
  `27.81%/24.23%`；
- `odometer_opt=True` 保留：device P50/P99 中位改善 `3.44%/5.57%`；
- `tile_align=8` 保留：device P50 基本持平（改善 `0.19%`），P99 回退 `1.87%`，属于本轮噪声域；
- `unify_block` 与 `pad_min_block_to_8` 在当前 greedy allocator 中不可生效，登记为配置清理项，
  不恢复已删除的 legacy inner-loop 路径；
- 无正确性缺陷和有证据的默认配置性能回退，不产生产品源码修改、不构建新 wheel。

最终状态：
`verified-active-defaults-retained-two-configs-ineffective-cleanup`。当前不提交、不推送。

## 2. 审计范围与安装态

| 配置 | 默认值 | 运行态可达性 | 结论 |
|---|---:|---|---|
| `reduce_xblock_by_input_stride` | True | 有效；改变 reduction free-axis greedy 顺序 | 保持开启 |
| `unify_block` | True | 当前无效；greedy axes 无条件并入 `_unify_names` | 仅跟踪清理 |
| `pad_min_block_to_8` | True | 当前无效；greedy/scalar axes 均被 padding 跳过 | 仅跟踪清理 |
| `odometer_opt` | True | 有效；改变 pid 累积块分解并消除 singleton | 保持开启 |
| `tile_align` | 8 | 有效；eligible contiguous real-block 对齐到 8 | 保持 8 |

运行态使用 P-022 独立安装环境
`/home/z50063656/Pass/venvs/p022-installed`，安装态 `npu_header.py` SHA256 为
`6f9d3a9277e61d0073f02012ce4e3a0c234ebd93e69dcdace80e20ac027b7dbe`。
所有测试从 `/home/z50063656/tmp` 启动，使用独立输出目录和 fresh process。

## 3. 正确性与结构证据

| 配置 | on/off workload | 数值结果 | 结构结果 |
|---|---|---:|---|
| input stride | dynamic permute + reduction | 2/2 | on 把 stride-1 的 `x1` 排为 tile0；off 使用输出 divisor 顺序 |
| unify | dynamic pointwise | 2/2 | on/off header trace 完全相同，确认配置无效 |
| padding | small pointwise | 2/2 | on/off header trace 完全相同，确认配置无效 |
| odometer | 四轴 pointwise | 2/2 | on 使用 cumulative block，off 使用连续 `// blocks` 链 |
| align | 1887-element pointwise | 2/2 | on 生成 align-8 real-block，off 保持原 tile |

合计 10/10。input-stride 两个动态 shape 均与 eager 精确相等；odometer-off 的最大绝对误差
为 `1.192093e-07`，满足 FP32 容差；其余 selected case 最大绝对误差均为 0。

`unify_block` 无效的直接原因是 `_is_unify_candidate()` 虽读取开关，但随后
`_unify_names |= set(_balanced_tile)` 无条件覆盖所有非 scalar greedy axes。padding 分支又显式
跳过 `_balanced_tile` 和 `_scalar_odo_names`，因此当前合法 free axis 没有进入最小 8 元素
padding 的路径。这里不为“让开关看起来有效”而恢复已删除的 legacy 分支，因为旧分支已有
real-block/tile 不一致的正确性风险。

## 4. 三轮 paired 性能

每个有效开关使用 on/off 各三轮 fresh process，warmup 10/runs 100。表中百分比为 on 相对 off；
负值表示开启更快。Event device 时间作为主性能依据，host P99 原样保留监控。

| 配置 | 指标 | off 中位 | on 中位 | on 相对 off |
|---|---|---:|---:|---:|
| input stride | device P50 | `0.290700 ms` | `0.209860 ms` | `-27.81%` |
| input stride | device P99 | `0.310900 ms` | `0.235580 ms` | `-24.23%` |
| input stride | 编译首跑 | `19.626 s` | `20.190 s` | `+2.87%` |
| input stride | peak | `2,447,872 B` | `2,447,872 B` | `0%` |
| odometer | device P50 | `0.225980 ms` | `0.218210 ms` | `-3.44%` |
| odometer | device P99 | `0.250560 ms` | `0.236600 ms` | `-5.57%` |
| odometer | 编译首跑 | `19.610 s` | `19.227 s` | `-1.95%` |
| odometer | peak | `6,850,048 B` | `6,850,048 B` | `0%` |
| align-8 | device P50 | `0.210950 ms` | `0.210540 ms` | `-0.19%` |
| align-8 | device P99 | `0.235420 ms` | `0.239820 ms` | `+1.87%` |
| align-8 | 编译首跑 | `18.604 s` | `18.931 s` | `+1.76%` |
| align-8 | peak | `40,448 B` | `40,448 B` | `0%` |

input-stride 的三轮 device P50 全部同向改善，是本组最明确收益。odometer 的中位收益较小，
但没有设备时间或内存回退证据。align-8 的 device 中位差异约 2% 内，不能据此推翻其 32-byte
访问对齐目的，保持默认值并在后续真实模型 cohort 继续观察。

## 5. 失败分类与证据

- 第一版 odometer probe 把被 broadcast 约束为 4 的 axis 标成 dynamic，on/off 均触发
  `ConstraintViolationError`；修正 probe 后 2/2 通过，属于测试构造失败。
- `perf_align_r2_off` 在物理设备 6 遇到 CANN `507033` TSD subprocess startup timeout；健康
  设备上的 `perf_align_r2_off_retry_v2` 通过并替代该样本，原失败证据保留。
- 性能矩阵脚本因保留上述环境失败最终 exit 1；选定有效矩阵本身为 18/18。

证据入口：

- 汇总：`results/t065_header_tiling_20260828/t065_summary.json`
- 运行探针：`t065_header_tiling_runtime_probe.py`
- 正确性脚本：`run_t065_header_tiling_matrix.sh`
- 性能脚本：`run_t065_header_tiling_performance.sh`
- 环境失败替代脚本：`run_t065_align_r2_off_retry.sh`
- 聚合脚本：`aggregate_t065_header_tiling_results.py`

## 6. 下一步

继续 T-066 / TE-CG-009 `group_dispatch` 与 `all_blocks_parallel`。T-065 没有新增待推送产品
代码；现有 P-014/P-016/P-017/P-018/P-019/P-020/P-021/P-022 继续留在各自本地隔离分支，
待剩余 pass 全部本地测试和优化结束后统一安排提交与推送。
