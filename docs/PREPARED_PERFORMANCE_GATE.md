# T-081～T-083性能准备与复核门禁

> 更新时间：2026-09-07 22:36 CST（UTC+08:00）

三个批次的worker均已实现但尚未在GPU/NPU运行。社区原生结构测试有效不能替代数值执行证据。
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

功能阶段完成后，控制节点复核生成gate。该JSON需包含以下内容，缺项就拒绝实测；当前没有签发有效gate，不提供填写为true即可绕过的模板：

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

获有效gate后，单个单元用下列入口（路径由控制节点实际产出填写）：

```bash
# 在Pass环境中运行；不是现在要求用户执行的GPU reference步骤
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_prepared_performance.py \
  --task T-081 --unit convert --device npu --gate /path/to/reviewed-convert-gate.json
```

T-083同一入口内部使用`python -m torch.distributed.run --standalone --nproc-per-node=2`，要求至少两张可见设备；用户无需再包一层torchrun。
必须在可控、无竞争条件下做性能测量；GPU reference默认共享策略不等于性能策略。

worker先检查精确改图/数值/stride与alias，通信还检查3→1或2→1的collective个数，然后才计时。
六个fresh-process按OFF1、ON1、ON2、OFF2、OFF3、ON3运行。每臂10次warmup和100次采样，保存host/Event原始样本、各rank显存峰值、编译耗时、PID、启动时间、实际加载源码哈希及软件版本。
分布式barrier放在计时之外；聚合按同sample取各rank最大值，再计算分位数并对三轮取中位数。
超时只终止本次创建的独立process-group，回收worker；不会扫描或终止其他GPU/NPU用户进程。

聚合器默认结论`PENDING_SOURCE_AND_KERNEL_REVIEW`。常量/视图可在下游同样消除，设备Event接近零时不能自动据比值报加速；仍需检视FX、生成代码与计时范围，再填写正式性能结论。
