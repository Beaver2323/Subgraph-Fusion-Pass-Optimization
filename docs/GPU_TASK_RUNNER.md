# GPU 指定任务一键执行说明

> 更新时间：2026-09-06 06:43 CST（UTC+08:00）
> 适用环境：`/data/z50063656` 下已安装的 PassGPURef、CUDA 12.6 与冻结 PyTorch source
> 当前任务：`T-076`、`T-077`、`T-078`、`T-079`、`T-080`

## 1. pull 后一键运行

以下命令在 GPU 服务器执行。先更新 tracker；仓库实际位置不同时只修改 `TRACKER_ROOT`：

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main
```

pull 成功后，指定任务和物理卡即可。以 T-078、物理 GPU 2 为例：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2
```

`2` 是物理卡编号示例。GPU 功能 reference **默认共享运行**：在 DEFAULT 计算模式下，
已有计算进程不阻止启动，也不要求 GPU 利用率归零；默认只要求至少 1024 MiB 空余显存。
无需手工激活虚拟环境，也无需查找本轮时间戳。

入口会自动完成：

1. 进入 `/data/z50063656/tmp`；
2. 激活 `/data/z50063656/envs/PassGPURef`；
3. 设置 CUDA、cuDNN、pip/Triton/Inductor cache；
4. 先执行该任务的零设备静态校验；T-078～T-080 同时检查功能计划、性能计划和中文 case guide；
5. 按共享/独占策略检查指定物理 GPU；指定 `--wait-gpu` 时等待条件满足；
6. 执行任务对应的原生 community suite；
7. 自动取得本轮 `reference-<timestamp>` 目录；
8. 自动生成包含 FX、日志、生成代码和 IR 原文的 1.1 文本 handoff，并建立不含时间戳的 `latest` 入口。

脚本不会安装或升级驱动、CUDA、Python、PyTorch，也不会修改 GPU 上的源码。
本入口只运行 GPU 功能 reference，不自动运行 NPU 或性能测试。
重跑 T-076/T-077 会生成新一轮基线，不替代旧日志的历史复核。

### 1.1 共享、独占与快速等卡

日常功能测试推荐直接使用共享 + 等卡，默认每 1 秒重新检查；显存满足门槛就启动，不等其他计算进程退出：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2 --wait-gpu
```

需要启动时没有其他计算进程，显式选择独占策略并等待：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2 --exclusive --wait-gpu
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--exclusive` | 不启用，即共享 | 启用后要求启动时没有计算进程，并排斥使用同一协作锁的其他 tracker 运行 |
| `--wait-gpu` | 不启用 | 指定卡不满足条件时轮询；不加则立即退出 3 |
| `--wait-timeout SEC` | `0` | `0` 一直等；正整数限制等卡总时长，需同时指定 `--wait-gpu` |
| `--poll-interval SEC` | `1` | 每轮未满足条件后的检查间隔，可选 1～60 秒，需同时指定 `--wait-gpu` |
| `--min-free-memory-mib MIB` | `1024` | 启动显存门槛；`0` 取消显存门槛，不保证测例不会 OOM |

例如最多等 2 小时、要求至少 4 GiB 空余显存：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2 --wait-gpu --wait-timeout 7200 --min-free-memory-mib 4096
```

若希望直接尝试、连默认 1 GiB 显存门槛也不等，可设为 `0`：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2 --wait-gpu --min-free-memory-mib 0
```

显存门槛只是启动瞬间的检查，不是显存预留或测例峰值估算。共享可能变慢或 OOM，功能测试中的
autotune 也可能受竞争影响；失败仍如实记录，不降低输入规模、不把 OOM/skip 改记为通过。
本入口的耗时不能作为正式性能收益证据；NPU `triton_experimental` 的同后端性能对比规则不变。

控制台每轮打印 `gpu_wait=queued` 及原因（显存不足、独占策略下有进程或协作锁忙）；
按 `Ctrl+C` 可取消，条件满足打印 `gpu_preflight=ready` 后自动进入测试。
只等待 `--gpu` 指定的一张卡，不自动换卡，也不是 FIFO 集群队列。

协作锁位于 `/data/z50063656/tmp/tracker-gpu-locks/gpu-2.lock`：共享运行可以并行，独占运行与
使用相同锁目录的其他 tracker 运行互斥。进程退出后内核释放锁，锁文件保留，不应手动删除。
它不能阻止未使用该入口的外部作业随后启动，因此 `--exclusive` 不是整机调度器级独占保证。
机器本身若为 EXCLUSIVE_PROCESS 模式且已有计算进程，共享选项也不能绕过它；脚本不杀其他
进程、不更改计算模式或驱动。[NVIDIA 计算模式说明](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)

运行策略与检查值写入 `environment.json` / 文本 handoff 的 `selected_environment`：
`PASS_GPU_EXECUTION_MODE`、`PASS_GPU_MIN_FREE_MEMORY_MIB`、
`PASS_GPU_PREFLIGHT_FREE_MEMORY_MIB`、`PASS_GPU_COMPUTE_MODE`。

## 2. 不再手工寻找 timestamp

任一任务完成后固定使用相应结果目录；与上面的 T-078 示例对应：

```bash
export RESULT_ROOT=/data/z50063656/tmp/t078-reference-results

ls -l "${RESULT_ROOT}/latest"
ls -l "${RESULT_ROOT}/latest-text-handoff.json"
python -m json.tool "${RESULT_ROOT}/latest-text-handoff.json" >/dev/null
sha256sum "${RESULT_ROOT}/latest-text-handoff.json"
```

其中：

- `latest` 指向本轮实际 `reference-<timestamp>` 目录；
- `latest-text-handoff.json` 固定指向 `latest/text-handoff.json`；正常为本轮可恢复原文的 1.1 文本交接文件；
- 控制台仍打印真实 `run_dir=` 和 `text_handoff=`，便于审计；
- 新一轮执行会原子更新软链接，不删除旧的带时间戳结果。

并行运行的自动目录名带随机后缀，避免同一秒启动相同任务时冲突。并行完成时，`latest` 指向
最后发布结果的运行，不保证是最后启动的运行；指定轮次请用控制台打印的真实 `text_handoff=`。
等卡、取消或启动前检查失败时没有产生新一轮结果，也不会更新 `latest`，勿把旧入口当成本轮结果。

2026-09-06 02:55 CST（UTC+08:00）修正：一键入口显式允许导出器**新建** run 内保留文件
`text-handoff.json`，解决原先入口路径与导出器保护规则冲突的问题。其他 run 内文件、已有输出或
软链接均不能覆盖；单独调用导出器时默认仍要求输出在原始 run 外。

若 runner 没有返回 artifacts 或文本导出失败，入口仍更新到**本轮失败状态**，其中
`handoff_status=export-failed`、`reference_valid=false`。复制这个 JSON 并附启动日志即可；
该状态文件不是有效 reference 证据。旧运行目录与旧文本保留，不会删除；半写入文件改名保留。

功能 reference 会清除继承的 `DO_PERF_TEST`/`USE_LARGE_INPUT`，避免误跑社区大规模性能分支。
部分 skip、expected failure、实际测试数与选定方法数不一致，都会阻止 valid reference。

T-076～T-080 分别对应 `t076-reference-results`～`t080-reference-results`，规则相同。

### 2.1 复制 JSON 到控制节点的固定接收目录

GPU 侧只有文本复制条件时，打印本轮完整 JSON：

```bash
cat /data/z50063656/tmp/t078-reference-results/latest-text-handoff.json
```

复制从开头 `{` 到最后 `}` 的完整 JSON 文本，不复制软链接本身，也不要混入提示符或启动日志。
将内容保存到当前 NPU 控制节点（Agent 能访问的机器）的对应任务目录：

```text
/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/
├── T-076/text-handoff.json
├── T-077/text-handoff.json
├── T-078/text-handoff.json
├── T-079/text-handoff.json
└── T-080/text-handoff.json
```

接收目录已通过 `.gitkeep` 纳入仓库，clone/pull 后会出现 T-076～T-080 文件夹；
见[接收目录说明](../results/incoming/README.md)。上面的 JSON 文件由用户粘贴生成，不提供空模板。
例如 T-078，更新后直接用编辑器粘贴保存即可；旧 checkout 尚未更新时可先手工创建目录：

```bash
mkdir -p /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/T-078
```

保存后应在控制节点运行 1.1 完整性校验，而不只检查 JSON 语法：

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/import_reference_text.py \
  --input /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/T-078/text-handoff.json \
  --validate-only
```

通过 GitHub 网页更新同名文件时，旧版本由 Git 历史保留；同一 checkout 需要并存多轮时另存带
run ID 的文件。保存后告知 Agent 任务号、路径和 commit，由 Agent 拉取复核；
接收目录不是自动验收入口，JSON 能解析也不代表测试通过。
接收说明和目录占位纳入 Git，`results/incoming/` 中实际 JSON/日志仍默认忽略；
不要用它覆盖 `results/current/` 的正式结果，
也不要把 GPU 回传文件放进 `results/audits/`（该目录存放控制节点生成的复核记录）。

当前一键入口默认生成 1.1 原文 handoff，可恢复已登记的 UTF-8 FX、日志、生成代码和常见 IR；
二进制只登记哈希和缺项原因。旧版 1.0 紧凑 handoff 只有摘要与哈希，不能恢复 FX 正文。
导出、复制、接收端校验、安全恢复和 FX 查看命令见
[GPU 原文 handoff 指南](GPU_TEXT_HANDOFF.md)。

## 3. 任务选择

完整运行时选择下面对应的一条命令，不需要把五批全部重跑：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-076 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-077 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-079 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-080 --gpu 2
```

目前仅 T-076～T-080 有可执行入口；T-081 起仍是任务草案，不能直接传给该脚本运行。

只做静态校验，不需要 GPU 编号：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --validate-only
```

只重跑一个 case：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-077 \
  --gpu 2 \
  --case REF-decompose-addmm-dynamic-native
```

任务映射固定为：

| 任务 | 内部 runner | 结果目录 |
| --- | --- | --- |
| `T-076` | `scripts/run_reference_all.sh` | `/data/z50063656/tmp/t076-reference-results` |
| `T-077` | `scripts/run_t077_reference_all.sh` | `/data/z50063656/tmp/t077-reference-results` |
| `T-078` | `scripts/run_t078_reference_all.sh` | `/data/z50063656/tmp/t078-reference-results` |
| `T-079` | `scripts/run_t079_reference_all.sh` | `/data/z50063656/tmp/t079-reference-results` |
| `T-080` | `scripts/run_t080_reference_all.sh` | `/data/z50063656/tmp/t080-reference-results` |

## 4. 路径覆盖

默认路径与当前 GPU 服务器一致。如目录发生变化，可在命令前覆盖：

```bash
export PASS_GPU_DATA_ROOT=/data/z50063656
export PYTORCH_ROOT=/data/z50063656/src/pytorch
export PASS_GPU_VENV=/data/z50063656/envs/PassGPURef
export CUDA_HOME=/data/z50063656/cuda-12.6
export PASS_TRACKER_WORK_DIR=/data/z50063656/tmp
```

实际测试始终由脚本从 `PASS_TRACKER_WORK_DIR` 发起，不会在 PyTorch 或 torch_npu 源码树中导入
`torch`。

## 5. 失败处理

- 默认共享时，已有计算进程不算失败；显存不足或协作锁忙且未指定 `--wait-gpu` 时退出 3；
- `--exclusive` 或机器 EXCLUSIVE_PROCESS 模式下，已有计算进程也会触发等待/退出 3；
- 等卡超时退出 124；等卡时 `Ctrl+C` 退出 130，终止信号退出 143；查询失败/超时或参数错误退出 2；
- 环境、源码 commit 或计划不一致时，静态校验/runner 直接失败；
- suite 失败时仍保留本轮结果并尝试生成文本 handoff；
- 脚本最终返回原 runner 退出码，不能把失败、skip 或 no-tests 当成成功；
- suite 已启动并发布结果后，将 `latest-text-handoff.json` 的完整文本复制回 NPU 控制节点；启动前失败则回传控制台错误。

T-078～T-080 的 reference wrapper 会先打印 `prepared_task_validation=OK`。这只表示测例来源、
单元覆盖、性能合同与讲解文档齐全，不表示 GPU/NPU 已执行，也不解锁 NPU 性能测试。
