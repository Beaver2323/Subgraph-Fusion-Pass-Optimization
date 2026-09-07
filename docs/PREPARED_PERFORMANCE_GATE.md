# T-081～T-083 性能执行与复核门禁

> 更新时间：2026-09-08 03:17 CST（UTC+08:00）
> 状态：NPU `triton_experimental` 功能 7/7、正式性能处置 7/7 完成；本页同时保留可复现入口和证据门禁。

三个批次已完成 GPU reference、NPU 数值/改图和性能实测。社区原生结构测试仍不能替代 NPU
数值执行证据，重新测量也必须先使用仓库中已签发且与当前源码哈希一致的 gate。
`run_prepared_performance.py --validate-only`只核对源文件、合同与可用目标，不导入torch。

```bash
# 控制节点，从规定tmp目录启动零设备准备校验
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_prepared_performance.py --task T-081 --validate-only
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_prepared_performance.py --task T-082 --validate-only
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_prepared_performance.py --task T-083 --validate-only
```

可选worker目标：T-081为`constant-fold`/`convert`，T-082为`permute`/`view`，T-083为`all-gather`/`all-reduce`/`reduce-scatter`。
每个单元的图、shape、意图、正负例和计时范围见对应T的功能性能指南；均由社区功能原图派生，未声称复用独立社区benchmark。

功能阶段完成后，控制节点复核生成 gate。当前 gate 位于
`results/current/T-081～T-083/performance_gates/`；该 JSON 必须包含以下内容，缺项或源码哈希变化就拒绝实测：

| 字段 | 校验 |
| --- | --- |
| task_id / acceptance_unit_id / measurement_workload | 精确绑定当前单元和`<worker_unit>-community-shape` |
| backend / pytorch_commit | NPU必须`triton_experimental`；PyTorch固定`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| input_spec / worker_sha256 | 输入shape/dtype与当前worker源码sha256吻合；输入要求连续 |
| correctness / target_rewrite | 必须分别为`passed`和`confirmed`；原件也逐字段校验 |
| graph_breaks / fallbacks / product_disabled | 分别为0/0/false；产品明确disable时只记录免测 |
| world_size | 前两批1；T-083必须2，真实NCCL/HCCL |
| reviewed_at / reviewer | 复核时间与执行复核者 |
| gpu_reference / target_functional | 各自为path+sha256；重新读取原件校验，GPU suite必须有效且含本单元 |

target_functional原件必须额外含`numerical_execution=true`、实际源文件绝对路径到sha256的`source_files`，以及与gate相同的输入/后端/改写/正确性字段。
T-083还需`process_group_backend=nccl/hccl`；FakeTensor/fake PG/world_size=1全部不能进入真实通信性能。
参数或源码变更会让旧gate失效，需要重新审核。

## 一键执行

所有 NPU 命令都从 `/home/z50063656/tmp` 启动。功能整批入口会自动激活 `Pass` 环境，且在导入
`torch` 前固定 `triton_experimental`：

```bash
cd /home/z50063656/tmp

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t081_t083_npu_function.sh \
  --task T-081 \
  --npu 0

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t081_t083_npu_function.sh \
  --task T-083 \
  --npu 0,1
```

单个性能单元统一用以下入口；脚本自动选择已复核 gate，并通过全局 `flock` 禁止 tracker 性能任务
并发污染 host/Event 尾延迟：

```bash
cd /home/z50063656/tmp

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_npu_performance_task.sh \
  --task T-081 \
  --unit convert \
  --npu 0

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_npu_performance_task.sh \
  --task T-083 \
  --unit all-reduce \
  --npu 0,1
```

T-083同一入口内部使用`python -m torch.distributed.run --standalone --nproc-per-node=2`，要求至少两张可见设备；用户无需再包一层torchrun。
必须在可控、无竞争条件下做性能测量；GPU reference默认共享策略不等于性能策略。

worker先检查精确改图/数值/stride与alias，通信还检查3→1或2→1的collective个数，然后才计时。
六个fresh-process按OFF1、ON1、ON2、OFF2、OFF3、ON3运行。每臂10次warmup和100次采样，保存host/Event原始样本、各rank显存峰值、编译耗时、PID、启动时间、实际加载源码哈希及软件版本。
分布式barrier放在计时之外；聚合按同sample取各rank最大值，再计算分位数并对三轮取中位数。
超时只终止本次创建的独立process-group，回收worker；不会扫描或终止其他GPU/NPU用户进程。

聚合器的原始结论必须经源码和生成 kernel 复核。正式结果是：T-081 `PERF_IMPROVED` 1、
`PERF_NEUTRAL` 1；T-082 `PERF_NEUTRAL` 2；T-083 `PERF_REGRESSED`、`PERF_IMPROVED`、
`PERF_MIXED` 各 1。详细数据与解释见
[T-081～T-083 NPU 功能、适配与性能报告](../report/t081_t083_npu_function_performance_20260908.md)。
