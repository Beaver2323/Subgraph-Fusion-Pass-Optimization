# T-078 GPU/reference Runner 操作说明

> 更新时间：2026-09-06 06:43 CST（UTC+08:00）
> 状态：已收到 12/12 通过的 1.0 紧凑摘要；尚需 1.1 原文 handoff 完成 GPU FX/日志正文复核
> 原则：先运行冻结 PyTorch commit 中的原生社区测例；direct 失败只回传证据，不在 GPU 机器临时写 adapter

2026-09-04 已按 PyTorch `copy_tests` 的真实命名规则，将两个 addcdiv 执行入口纠正为带
`_cuda` 后缀的方法；请先 pull 包含该修正的版本再运行。

## 1. 一键执行

GPU 服务器 pull 最新 `main` 后执行：

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2
```

将 `2` 替换为所选物理 GPU 编号；功能 reference 默认共享，允许已有计算进程。
加 `--wait-gpu` 每 1 秒检查启动条件，加 `--exclusive` 才要求启动时无计算进程。
入口会自动进入 `/data/z50063656/tmp`、激活 `PassGPURef`、检查 PyTorch commit/工作树、
按所选策略检查 GPU、执行 12 个 fresh-process cases，并生成 1.1 原文文本 handoff。
显存门槛、等卡超时和固定结果入口见[通用一键说明](GPU_TASK_RUNNER.md)。

## 2. 预期静态校验

不使用 GPU 时可先检查计划：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --validate-only
```

预期：

```text
prepared_task_validation=OK task=T-078 units=4 cases=12 variants=20 performance_units=4 guide=valid
reference_plan_validation=OK acceptance_units=4 cases=12 community_tests=12 variants=20 executed_variants=20 non_executed_variants=0
torch_imported=0 gpu_executed=0
gpu_task_validation=OK task=T-078
```

静态校验会同时检查 `upstream/t078_performance_plan.yaml` 与中文
`docs/T078_FUNCTION_PERFORMANCE_GUIDE.md`，但不会运行性能。后者解释每个功能/性能测例的来源、
输入和判据。

## 3. 单 case 重跑

例如只重跑 addcdiv codegen：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2 \
  --case REF-addcdiv-fma-codegen-native
```

可用 case ID：

```text
REF-addcdiv-fma-bitwise-native
REF-addcdiv-fma-codegen-native
REF-partial-reuse-positive-native
REF-partial-reuse-negative-native
REF-unfuse-addmm-core-native
REF-unfuse-addmm-accumulator-negative-native
REF-unfuse-addmm-leaf-native
REF-unfuse-addmm-expanded-native
REF-unfuse-addmm-half-preserve-native
REF-unfuse-addmm-half-enabled-native
REF-unfuse-baddbmm-core-native
REF-unfuse-baddbmm-alpha-beta-native
```

## 4. 结果与文本回传

```bash
export RESULT_ROOT=/data/z50063656/tmp/t078-reference-results
export RUN_DIR="$(readlink -f "${RESULT_ROOT}/latest")"
export TEXT_HANDOFF="$(readlink -f "${RESULT_ROOT}/latest-text-handoff.json")"

python -m json.tool "${TEXT_HANDOFF}" >/dev/null
sha256sum "${TEXT_HANDOFF}"
```

请复制 `latest-text-handoff.json` 的完整文本。12/12 cases 均 `passed` 且 `reference_valid=true` 才能
冻结 T-078；失败、skip、no-tests 或 FX artifacts 缺失都必须原样保留，不能记作 PASS。
新版一键入口默认携带可恢复的 FX、日志、生成代码和常见 IR 原文；旧 1.0 文件不能恢复正文。
完整校验、手工重导出旧 run、GitHub 文本复制与恢复说明见
[GPU 原文 handoff 指南](GPU_TEXT_HANDOFF.md)。

## 5. 环境边界

- PyTorch 必须为 `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` 且工作树 clean；
- 使用既有 `/data/z50063656/envs/PassGPURef`、CUDA 12.6.3 和 A100 环境，不重新安装；
- 所有测试由脚本从 `/data/z50063656/tmp` 启动；
- GPU reference 使用社区默认 Inductor/CUDA 合同；NPU 阶段才固定使用 `triton_experimental`；
- GPU 机器只生成 reference，不修改 torch_npu 产品代码。
