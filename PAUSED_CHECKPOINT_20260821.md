# Pass NPU 审计暂停检查点（2026-08-21）

## 暂停状态

- 状态：`paused-by-user`。
- 恢复记录：用户随后于 2026-08-21 明确要求继续；本文件保留为历史恢复基线，当前执行状态与新证据以 `change_control.md:E-030` 起的记录为准。
- 已停止：新的 NPU 正确性实验、benchmark、profiler、源码接入、wheel 构建和环境安装。
- 最后完成：T-014 standalone Triton 的 shape-A/128³ 双累加器正确性重试。
- 尚无结论：T-014 未验证 unaligned、单 task 数、稳态 p50/p99、编译对照和峰值内存，不能声称性能提升，也不能改变 `mm_plus_mm` 的最终 verdict。

## 固定环境

| 项目 | 暂停时状态 |
|---|---|
| 启动入口 | `/home/z50063656/Pass/activate_pass.sh` |
| 测试 cwd | 必须从 `/home/z50063656/tmp` 启动；不得在 torch_npu 源码目录内 import torch |
| Python | 3.11.15，Conda `Pass` (`/home/z50063656/envs/Pass`) |
| PyTorch | `release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc`，源码构建 wheel，以 `--no-deps --force-reinstall` 安装 |
| 当前 wheel | `Pass/src/torch_npu/dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl` |
| wheel SHA256 | `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704` |
| Triton Ascend | `release/3.2.2@8bd9f380d2786002b84b5248f00838c26f900515` |
| CANN / NPU | CANN 9.0.1；8 x Ascend910B2 |

## 已完成里程碑

1. T-011：补齐 torch_npu `make_reduction(..., strict_sum=...)` 接口；重建/安装 wheel；最小 sum、addmm vector/full-bias backward、inference 和 mm_plus_mm backward 回归通过。addmm 最终 verdict 为 `supported-beneficial`。
2. T-012：different-K 当前 fallback 的 shape-A/unaligned paired baseline 完成，p50 分别为 -0.30%/+2.53%，结论为 neutral。
3. T-013：NPU profiler 确认每次为两个 `aclnnMm` 加一个 Triton add；理论端到端上限为 17.68%/16.02%，candidate 单 task 预算约为 33.93/31.13 μs。
4. T-014：初版 split-store 大面积错误和单 accumulator 的 4 个舍入差异均已定位。双 fp32 accumulator 分别 cast fp16 后相加的 shape-A/128³ candidate 为 `compile-correct`，最大/平均绝对误差均为 0。

T-014 最终证据：
`results/t014_mmplus_different_k_triton_20260821/shape_a_two_acc_retry2/result.json`。
baseline 首次编译+执行为 `19906.98771 ms`，candidate 为 `5107.895 ms`；两者只是单次编译诊断值，不能用作性能比较。

## 暂停时工作树

- PyTorch：tracked/untracked 状态为空。
- torch_npu：唯一 tracked 功能 diff 为
  `torch_npu/_inductor/lowering.py` 的 T-011 `strict_sum` 修复；既有 wheel/codegen 构建产物保留，不清理、不提交。
- Triton Ascend：保留进入本任务前已有的三个 tracked 修改：
  `third_party/ascend/backend/backend_register.py`、
  `third_party/ascend/backend/runtime/__init__.py`、
  `third_party/ascend/backend/utils.py`。本任务未覆盖或回退它们。
- 审计目录：T-011 至 T-014 的脚本、结果、报告和文档均原样保留；T-014 仍是 standalone，不在 Inductor registry/lowering 中注册。
- 评估矩阵：251 条，247 `not-run`、3 `unsupported`、1 `supported-beneficial`；`mm_plus_mm` 最终 verdict 仍为 `not-run`。

## 设备和进程收尾

- T-014 worker 已正常返回 `functional-complete`，执行会话已经结束，没有继续启动本任务 worker。
- 收尾 `npu-smi info` 仍在物理 NPU 6 显示 PID 2314011、约 4038 MiB；随后的 `/proc/2314011` 检查显示该主机 PID 已不存在，但第二次 `npu-smi` 仍保留该条目。没有终止、清理或推断该进程归属。
- NPU 0、2–5、7 存在其他 Python 任务。恢复时不得沿用历史卡号；必须重新运行 `npu-smi info`，只选择进程表明确为空的物理卡。

## 精确恢复点

恢复工作时先阅读本文、`current_status_and_background.md` 和
`change_control.md:E-029/T-014`，然后按以下顺序继续：

1. 只读复核环境、wheel SHA、三个源码工作树和 NPU 进程表。
2. 在 fresh 结果目录运行同一 128³ candidate 的 unaligned fp16 correctness；不同时启动 benchmark。
3. 只有 unaligned 达到既定 `rtol=atol=0.01`，才用 profiler 验证 candidate 每迭代恰好一个 NPU task，并检查单 task 是否低于约 31.13 μs。
4. profile 通过后才执行 current fallback/candidate 的 warmup 10、runs 100、3 轮交错 paired benchmark，记录 mean/stdev/p50/p99、首次编译和峰值内存。
5. 只有 shape-A 与 unaligned 的端到端 p50 都稳定改善超过 10%，才允许登记 dtype/layout/dynamic/backward 扩展；正式源码接入必须另立提案。

恢复时的第一个动态任务等价于：从 `/home/z50063656/tmp` 激活
`/home/z50063656/Pass/activate_pass.sh`，选择当时空闲设备，运行
`t014_mmplus_different_k_triton.py --shape-profile unaligned --tile-configs bm128_bn128_bk128`
并写入新的结果目录。该命令在本次暂停收尾中没有执行。

## 文档入口

- [当前状态与背景](current_status_and_background.md)
- [变更控制与逐次证据](change_control.md)
- [从头学习指南](inductor_pass_npu_beginner_guide.md)
- [T-012 paired baseline](report/t012_mmplus_different_k_baseline_20260821.md)
- [T-013 profiler 报告](report/t013_mmplus_different_k_profile_20260821.md)
