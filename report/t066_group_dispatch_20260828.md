# T-066 triton_experimental group/all-block dispatch 验证报告

## 1. 结论

T-066 / TE-CG-009 已完成。静态契约、experimental Inductor 正确性、65535/65536 边界、
all-block size sweep、三轮性能和 P-023 安装态回归全部通过既定判据。

- 非 A5 的 group-dispatch 代码生成保持开启：48-core 修正样本相对 backend auto size-1 的
  device P50/P99 中位改善 `3.15%/3.55%`，峰值内存相同；
- `group_dispatch` 配置快照没有消费者，on/off 生成相同 group loop；登记为死配置清理项，
  不为它发明未经验证的 off 路径；
- all-blocks-parallel 后端能力保持显式 opt-in，不全局设置
  `TRITON_ALL_BLOCKS_PARALLEL`；plain grid 65536 只有在该 gate 打开时合法；
- P-023 本地修复保留：后端 gate 关闭时不再生成会被折回 size 1 的无效
  `auto_blockify_size=2/4/8` 候选；gate 打开时三候选完整保留；
- P-023 wheel 构建、独立 venv 安装、目标 UT 3/3、近邻 2/2、candidate contract 两态、
  dynamic NPU smoke 和静态检查均通过。

最终状态：`verified-group-retained-dead-config-recorded-p023-local-fix`。当前不提交、不推送。

## 2. 配置与实现边界

| 项目 | 实际行为 | 结论 |
|---|---|---|
| `group_dispatch=True` | config 被 snapshot，但无消费者 | 死配置，后续仅做清理 |
| 非 A5 group codegen | prologue 和 group loop 无条件生成 | 保留 |
| `all_blocks_parallel=True` | 只允许 torch_npu 生成 auto-blockify 候选 | 不是 backend enable |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | Triton compiler 与 driver 的真实能力 gate | 维持用户显式 opt-in |
| P-023 双门控 | config 与 env gate 同时开启才生成 2/4/8 | 保留本地修复 |

运行时设备为 Ascend B2，实际探测 `48` 个 vector cores。源码历史注释中的“40 cores”不能
作为运行时常量；最初硬编码 40 的 group 性能样本只保留为探索记录，最终性能结论全部使用
动态探测的 48-core 重跑样本。

## 3. 静态、正确性与边界证据

静态契约验证了以下事实：

- `group_dispatch` 默认 true、快照存在、消费者为 0；group codegen 无条件存在；
- `all_blocks_parallel` 的 formula/legacy 候选路径可达；
- installed Triton 3.2.0 compiler、driver 都独立读取环境 gate，默认关闭；
- 40-core 与 48-core 下的 22 组 dispatch 算术均无遗漏、无重叠、覆盖精确。

experimental Inductor 正确性矩阵：

| workload | config on | config off | 结构结论 |
|---|---:|---:|---|
| underfill | PASS | PASS | 两态均有 48-core group loop |
| tail dynamic（两形状） | PASS | PASS | 两态均精确相等 |
| multi-axis | PASS | PASS | 最大绝对误差 `1.192093e-07` |

合计 6/6，且两态的生成代码都包含 `total_thread=48`、tail 分配与
`for i in range(group_size)`。这证明 config state 会变化，但实际 codegen 不变化。

边界证据：

- plain grid=65535、backend env off：PASS；
- plain grid=65536、backend env off：预期 EE1003 `coreDim <= 65535` 失败；
- plain grid=65536、backend env on：PASS；
- group grid=100003、backend env off、物理 grid=48：PASS 且结果精确。

## 4. 三轮性能与 all-block size sweep

workload 为 logical blocks 100003、block 8，warmup 10/runs 100。group 与 backend auto size-1
各三轮 fresh process；百分比为 group 相对 auto，负值表示 group 更快。

| 指标 | backend auto size-1 中位 | 48-core group 中位 | group 相对 auto |
|---|---:|---:|---:|
| device P50 | `0.409340 ms` | `0.396440 ms` | `-3.15%` |
| device P99 | `0.428420 ms` | `0.413220 ms` | `-3.55%` |
| 编译首跑 | `16.478 s` | `16.580 s` | `+0.62%` |
| peak | `16,018,432 B` | `16,018,432 B` | `0%` |

单轮 auto-blockify size sweep 均数值正确：

| size | device P50 | device P99 | P50 相对 size-1 |
|---:|---:|---:|---:|
| 1 | `0.409450 ms` | `0.517480 ms` | baseline |
| 2 | `2.776500 ms` | `2.825040 ms` | `+578.10%` |
| 4 | `2.655340 ms` | `2.755920 ms` | `+548.51%` |
| 8 | `2.611210 ms` | `4.973780 ms` | `+537.74%` |

这个 workload 上 size 1 明显最好，但不据此删除 2/4/8：当 backend 能力由用户显式开启时，
它们仍是合法 autotune 候选。P-023 只消除 backend gate 关闭时必然无效的重复候选。

## 5. P-023 修复与安装态验证

P-023 独立工作树：
`/home/z50063656/Pass/worktrees/torch_npu_p023_all_blocks_env_gate`。修改三个文件：

- `npu_triton_heuristics.py`：新增动态双门控 helper，formula/legacy 两条路径复用；
- `config.py`：澄清 config 只准入候选；
- `test_triton_experimental_autotune.py`：新增 backend-off/config-off 覆盖，并固定 on case 的 env。

wheel SHA256：
`41d67250693edf0d5c8dc213484a1a31025996343df72067778491983b36d879`。源码、wheel 内和
`/home/z50063656/Pass/venvs/p023-installed` 安装后的 heuristic/config 哈希完全一致。

| 验证项 | 结果 |
|---|---:|
| env=0/config-on → 无 auto 候选 | PASS |
| env=1/config-on → `[2,4,8]` | PASS |
| target UT | 3/3 PASS |
| 近邻 `numel_one` / `hole_grid` | 2/2 PASS |
| dynamic experimental Inductor 两形状 | 2/2 精确相等 |
| lintrunner 六类规则 | PASS |

dynamic smoke 首次没有带既有 `CPATH`，Triton launcher 报 `ATen/ATen.h` 缺失。按图模式规则
检查该次 `output_code.py`，确认 experimental wrapper、融合 kernel 和 48-core group loop 已
生成；使用基线矩阵相同的 `CPATH + CC wrapper + TRITON_DISABLE_PRECOMPILE=1` 重跑后通过。
因此该次归类为环境失败，不是产品回归。

## 6. 失败分类与证据入口

- `plain_65536_env_off_v1`：预期硬件/driver 边界失败，是能力 gate 证据；
- `group_41_smoke_v1`：probe 内嵌 `tl` helper 触发 NameError，修正 probe 后通过；
- 初始 group r1-r3：硬编码 40 cores，性能样本无效；48-core retry 三轮替代；
- P-023 第一次构建：sandbox 网络/子模块环境阻断；第二次构建：缺 op-plugin 本地子模块；
  本地同基线子模块补齐后第三次构建成功；
- P-023 第一次 dynamic smoke：缺少 CPATH，保留 `output_code.py` 与完整 traceback。

证据入口：

- 汇总：`results/t066_dispatch_20260828/t066_summary.json`
- 静态契约：`t066_dispatch_static_contract.py`
- 候选契约：`t066_candidate_probe.py`
- Inductor 探针：`t066_inductor_dispatch_probe.py`
- standalone 探针：`t066_standalone_dispatch_probe.py`
- 聚合脚本：`aggregate_t066_dispatch_results.py`
- P-023 四件套：`issues/P023_all_blocks_env_gate/`

## 7. 下一步

继续 T-067 / TE-AUTO-001。P-023 与此前 P-014/P-016/P-017/P-018/P-019/P-020/P-021/P-022
一样留在本地隔离工作树；等剩余 pass 全部完成本地测试与优化后，再统一整理提交/推送顺序。
