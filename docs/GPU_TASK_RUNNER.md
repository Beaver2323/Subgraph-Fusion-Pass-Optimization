# GPU 指定任务一键执行说明

> 更新时间：2026-09-04 09:08 CST（UTC+08:00）
> 适用环境：`/data/z50063656` 下已安装的 PassGPURef、CUDA 12.6 与冻结 PyTorch source
> 当前任务：`T-076`、`T-077`、`T-078`、`T-079`、`T-080`

## 1. pull 后一键运行

GPU 服务器更新 tracker 后，只需指定任务和空闲物理卡：

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2
```

入口会自动完成：

1. 进入 `/data/z50063656/tmp`；
2. 激活 `/data/z50063656/envs/PassGPURef`；
3. 设置 CUDA、cuDNN、pip/Triton/Inductor cache；
4. 先执行该任务的零设备静态校验；T-078～T-080 同时检查功能计划、性能计划和中文 case guide；
5. 检查指定物理 GPU 当前没有计算进程；
6. 执行任务对应的原生 community suite；
7. 自动取得本轮 `reference-<timestamp>` 目录；
8. 自动生成文本 handoff，并建立不含时间戳的 `latest` 入口。

脚本不会安装或升级驱动、CUDA、Python、PyTorch，也不会修改 GPU 上的源码。

## 2. 不再手工寻找 timestamp

任一任务完成后固定使用相应结果目录；例如 T-080：

```bash
export RESULT_ROOT=/data/z50063656/tmp/t080-reference-results

ls -l "${RESULT_ROOT}/latest"
ls -l "${RESULT_ROOT}/latest-text-handoff.json"
python -m json.tool "${RESULT_ROOT}/latest-text-handoff.json" >/dev/null
sha256sum "${RESULT_ROOT}/latest-text-handoff.json"
```

其中：

- `latest` 指向本轮实际 `reference-<timestamp>` 目录；
- `latest-text-handoff.json` 指向本轮完整文本交接文件；
- 控制台仍打印真实 `run_dir=` 和 `text_handoff=`，便于审计；
- 新一轮执行会原子更新软链接，不删除旧的带时间戳结果。

T-076～T-080 分别对应 `t076-reference-results`～`t080-reference-results`，规则相同。

## 3. 任务选择

完整运行：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-076 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-077 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-079 --gpu 2
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-080 --gpu 2
```

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

- GPU 已存在计算进程时，脚本退出码为 3，不抢占、不杀进程；
- 环境、源码 commit 或计划不一致时，静态校验/runner 直接失败；
- suite 失败时仍保留本轮结果并尝试生成文本 handoff；
- 脚本最终返回原 runner 退出码，不能把失败、skip 或 no-tests 当成成功；
- 将 `latest-text-handoff.json` 的完整文本复制回 NPU 控制节点即可。

T-078～T-080 的 reference wrapper 会先打印 `prepared_task_validation=OK`。这只表示测例来源、
单元覆盖、性能合同与讲解文档齐全，不表示 GPU/NPU 已执行，也不解锁 NPU 性能测试。
