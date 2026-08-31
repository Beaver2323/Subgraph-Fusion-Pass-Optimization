# T-076 GPU/reference Runner 人工操作说明

> 更新时间：2026-08-31 20:00 CST（UTC+08:00）
> 状态：runner 已完成静态校验，等待 GPU 机器执行原生社区测例。
> 核心规则：先取得 direct 结果；没有 direct blocker 证据，不创建或运行 adapter。

## 1. 执行前提

GPU 机器需要准备：

- 可用的 CUDA GPU 和 Triton；
- PyTorch source checkout 固定在
  `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`；
- PyTorch source working tree 为 clean；
- 当前 Python 环境中的 `torch.version.git_version` 与上述 source commit 完全一致；
- 能直接运行 PyTorch `test/inductor` 社区测试的依赖；
- tracker 仓库已拉取包含 T-076 runner 的最新提交。

runner 不安装依赖、不切换 Git commit、不修改 PyTorch，也不自动提交 artifacts。环境或 commit
不一致时会在生成 `environment.json` 后停止，避免建立伪 baseline。

## 2. 更新仓库并核对 commit

以下路径使用当前项目的标准目录；GPU 机器路径不同时，只替换 repo/source/output 的绝对路径，
测试工作目录仍必须位于源码树外。

```bash
cd /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git pull --ff-only origin main

git -C /home/z50063656/Pass/src/pytorch rev-parse HEAD
```

第二条命令必须输出：

```text
8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b
```

然后激活 GPU 机器已有的 PyTorch reference 环境。不要使用 NPU 的
`/home/z50063656/Pass/activate_pass.sh` 代替 GPU 环境。

## 3. 先做零设备静态校验

从 `/home/z50063656/tmp` 发起：

```bash
cd /home/z50063656/tmp
bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_reference_all.sh \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --validate-only
```

预期关键输出：

```text
reference_plan_validation=OK acceptance_units=5 cases=13 community_tests=13 variants=20 executed_variants=14 non_executed_variants=6
torch_imported=0 gpu_executed=0
```

这一步只校验 manifest、执行映射、community test 文件/方法、source commit 和 clean working
tree，不导入 `torch`。

## 4. 执行全部原生社区测例

结果应写到仓库外，避免原始 debug/cache 误入 Git：

```bash
cd /home/z50063656/tmp
bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_reference_all.sh \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --output-root /home/z50063656/tmp/t076-reference-results
```

执行合同如下：

1. 13 个 case 全部直接调用冻结 commit 中的社区测试方法；
2. 每个 case 使用 fresh Python process、独立 Inductor cache、独立 debug/trace 目录；
3. 某个 case 失败、skip 或超时后继续执行其余 case；
4. `TORCH_COMPILE_DEBUG` 和 `TORCH_TRACE` 用于保留 FX/IR/codegen 证据；
5. runner 不读取社区测试未打印的 counter 值，而是如实标记“由原测试内部断言验证”；
6. 首批没有配置 benchmark；只有 functional/reference artifact gate 有效后才能另行增加性能任务。

脚本在部分失败时最终返回非零，但 artifacts 和 summary 已保留，不代表批次中途丢失。

## 5. 单 case 重跑

先保留首轮完整批次，再用新的 run id 重跑，不覆盖旧证据：

```bash
cd /home/z50063656/tmp
bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_reference_all.sh \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --output-root /home/z50063656/tmp/t076-reference-results \
  --case REF-mm-plus-mm-native
```

可用 case ID 以 `upstream/reference_plan.yaml` 为准。不要直接修改社区图、shape、dtype、正负例
语义或 expected match 来让失败消失。

## 6. 结果目录

每次运行生成独立的 `reference-<timestamp>` 目录：

```text
reference-<timestamp>/
├── environment.json
├── environment_probe_stdout.log
├── environment_probe_stderr.log
├── manifest_snapshot.json
├── reference_plan_snapshot.json
├── reference_summary.json
├── reference_summary.md
└── cases/<case-id>/
    ├── metadata.json
    ├── reference_result.json
    ├── fx_before.txt
    ├── fx_after.txt
    ├── stdout.log
    ├── stderr.log
    ├── benchmark.json
    ├── artifact_inventory.json
    ├── debug/
    └── structured_trace/
```

`reference_valid=true` 同时要求原生测试通过并捕获计划要求的 FX before/after。只有测试返回 0、
但实际为 skip、未发现测试或缺失 FX artifact 时，仍不能作为有效 baseline。

## 7. 打包与回传

不要先删除失败 case 或大日志。退出码为非零也应打包整个 run 目录：

```bash
cd /home/z50063656/tmp/t076-reference-results
tar -czf reference-<timestamp>.tar.gz reference-<timestamp>
sha256sum reference-<timestamp>.tar.gz > reference-<timestamp>.tar.gz.sha256
```

回传内容为 `.tar.gz`、`.sha256` 和脚本最终 stdout/stderr。若团队选择通过 Git 交接，应先人工
检查体积、凭据和二进制，再决定是否只提交 summary 或使用独立 artifacts 存储；runner 不替人工
执行 `git add/commit/push`。

## 8. Direct blocker 与 adapter 决策

收到 artifacts 后由 NPU 控制节点逐 case 分类：

| Direct 结果 | 后续动作 |
| --- | --- |
| `passed` 且 `reference_valid=true` | 保留 direct，不创建 adapter |
| big-GPU/device harness skip | 先确认硬件是否符合 reference 合同，再评估只移除 harness 的薄 adapter |
| backend/device 常量阻塞 | 记录原 traceback 后，允许设计只注入 device/backend 的 adapter |
| 仅 artifact capture 缺失 | 保留原图和原断言，只补采集薄层 |
| 正确性、counter 或 FileCheck 断言失败 | 先按 upstream/environment failure 分析，不用 adapter 改预期 |
| 图、shape、dtype、正负例需改变才能通过 | 禁止称为最小 adapter，退回 mapping 审核 |

任何 adapter/extracted case 进入 `reference_plan.yaml` 前，都必须记录对应 direct case 的 blocker
artifact。首批当前没有 `adapters/` 目录，这是 direct 尚未实跑时的有意状态。
