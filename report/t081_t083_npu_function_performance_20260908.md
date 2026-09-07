# T-081～T-083 NPU 功能、适配与性能处置报告

> 更新时间：2026-09-08 03:12:20 CST（UTC+08:00）
> PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> NPU：Ascend 910B2，CANN 9.0.1，`triton_experimental`
> 结论范围：7 个已冻结 acceptance unit；延期候选仍不计数。

## 1. 结论

T-081～T-083 的 7 个单元已完成 GPU reference、NPU 数值/命中/改图和正式六臂性能处置。NPU 功能 7/7 通过，未修改 torch_npu 产品实现；本轮只修复 tracker 的观察、注册扫描、环境启动和性能互斥能力。

| T | 单元 | NPU 功能 | NPU Event p50 / p99 改善 | 性能结论 | 产品处置 |
| --- | --- | --- | ---: | --- | --- |
| T-081 | constant-fold-uniform-value | PASS，恒等图 1 kernel→0 | 29.42% / 25.57% | `PERF_IMPROVED` | 保留默认启用 |
| T-081 | pointless-convert | PASS，2 cast→1 cast | -0.04% / 0.82% | `PERF_NEUTRAL` | 不改配置 |
| T-082 | pointless-permute-pair | PASS，2 permute→0 | 3.78% / 8.98% | `PERF_NEUTRAL` | 最终均为零 kernel，不把微小时延当收益 |
| T-082 | pointless-view-pair | PASS，2 view→0 | 5.35% / -4.06% | `PERF_NEUTRAL` | 最终均为零 kernel，不改配置 |
| T-083 | bucket-all-gathers | PASS，HCCL 3→1 | -13.43% / -5.36% | `PERF_REGRESSED` | 保持上游默认 `none` |
| T-083 | bucket-all-reduce | PASS，HCCL 2→1 | 21.41% / 17.69% | `PERF_IMPROVED` | 默认仍为 `none`；进入真实模型候选复核 |
| T-083 | bucket-reduce-scatters | PASS，HCCL 2→1 | 6.05% / -10.32% | `PERF_MIXED` | 保持上游默认 `none` |

性能口径是社区功能正例派生的单次 compiled 子图调用，不是完整训练模型端到端。社区没有这 7 个单元的独立目标级 OFF/ON benchmark；因此复用社区图、shape、dtype，新增 fresh-process OFF/ON、host/NPU Event、显存与生成代码采集。

## 2. 执行链和门禁

joint-graph 单元的真实调用链：

```text
# torch/_inductor/compile_fx.py → fx_passes/joint_graph.py
torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
  → AOTAutograd joint graph
  → joint_graph_passes(gm)
  → constant_fold_uniform_value 或 early_patterns/patterns.apply
  → post_grad_passes
  → lowering → scheduler → triton_experimental codegen
  → NPU runtime
```

通信分桶单元的真实调用链：

```text
# torch/_inductor/fx_passes/post_grad.py:348-396
torchrun --nproc-per-node=2
  → torch.compile(..., npu_backend="triton_experimental")
  → post_grad_passes(gm)
  → bucket_reduce_scatter / bucket_all_reduce / bucket_all_gather
  → stable_topological_sort
  → lowering/codegen
  → HCCL collective + wait_tensor
```

功能阶段先运行 eager 数值参照，再在 fresh process 中编译。ON 必须捕获 handler 调用和实际图变化；OFF 必须是 handler 0 次。T-083 还必须是 `world_size=2`、HCCL，并逐 rank 验证 3→1 或 2→1。功能原件通过后，控制节点签发逐单元 gate；性能 worker 每个臂重新校验 gate、commit、源码哈希、输入和正确性。

正式性能顺序固定为 `OFF1→ON1→ON2→OFF2→OFF3→ON3`，每臂独立进程，10 次预热、100 次采样，三轮分位数取中位数。通信按同一 sample 取两 rank 最大值，barrier 在计时区间外。

## 3. T-081 pattern 解释与对照

### 3.1 均匀常量折叠

```python
# torch/_inductor/fx_passes/joint_graph.py:727
if config.joint_graph_constant_folding:
    GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
        constant_fold_uniform_value
    )
```

意图是把 `full(x.shape, 1) - 1` 识别为均匀零值，使 `x + 0` 直接返回 `x`。GPU 原生动态形状和自指 shape 测试通过；NPU OFF 保留 `full/sub/add`，ON 图直接返回输入，数值相同。生成代码从 1 个 Triton kernel 变为 0，NPU Event p50/p99 分别改善 29.42%/25.57%。

### 3.2 无意义转换链消除

```python
# torch/_inductor/fx_passes/joint_graph.py:842-864
def pointless_convert(match, arg, dtype1, dtype2):
    if arg_val.dtype in allowed and dtype1 in allowed and dtype2 in allowed:
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        node.replace_all_uses_with(repl)
        match.erase_nodes()
```

意图是在舍入安全时将 `fp16→fp32→fp16` 收敛成一次到最终 dtype 的转换；变窄中间类型、模拟精度不安全路径和整数往返仍由 guard 保留。GPU 的五组社区结构合同通过。NPU ON 捕获两处目标改图且数值正确，但 OFF 两次 cast 已被同一 Triton kernel 融合，ON 也是一个 kernel，所以设备时延中性。

## 4. T-082 pattern 解释与对照

### 4.1 互逆 permute

```python
# torch/_inductor/fx_passes/joint_graph.py:957-967
for i in range(rank):
    if perm1[perm2[i]] != i:
        return
node.replace_all_uses_with(arg)
match.erase_nodes()
```

两次 permutation 互为逆映射时复合为恒等；若中间结果有额外用户，社区负例要求保留。GPU 2D/3D 结构正负例通过。NPU ON 从两个 permute 变为直接返回输入，数值、stride 和 storage alias 均通过。

### 4.2 互逆 view

```python
# torch/_inductor/fx_passes/joint_graph.py:936-945
arg_size = list(arg.meta["val"].shape)
if definitely_equal(arg_size, size2):
    node.replace_all_uses_with(arg)
    match.erase_nodes()
```

第二个 view 恢复原 shape 时删除整个 pair，同时覆盖 `-1` 与 unbacked 动态形状保护。GPU 静态结构和动态数值例通过；NPU ON 也直接返回输入且保持 alias/stride。

这两个单元在目标 pass 层的 OFF/ON 图不同，但 NPU 最终代码都没有设备 kernel：OFF 中的元数据操作在更下游同样被消除。因此微小 Event/host 差异属于调用噪声，正式结论均为 `PERF_NEUTRAL`，不能据 3%～9% 比值宣称内核加速。

## 5. T-083 pattern 解释与对照

### 5.1 all-gather 分桶

```python
# torch/_inductor/fx_passes/post_grad.py:380-394
if config.bucket_all_gathers_fx != "none":
    GraphTransformObserver(gm, "bucket_all_gathers").apply_graph_pass(
        lambda graph: bucket_all_gather(graph.owning_module, ..., bucket_mode)
    )
```

三个独立 all-gather 被打平、拼接为一个 collective，再 split/clone 恢复三个输出。GPU world_size=1 只证明社区结构，不授权跨 rank 性能。NPU 使用真实双 rank HCCL，逐 rank 证明 3→1 且数值正确；但 `[64]` 小消息的 ON 路径新增打包和三个拆包 Triton kernel，Event p50/p99 回退 13.43%/5.36%。上游默认 `bucket_all_gathers_fx="none"` 保持不变。

### 5.2 all-reduce 分桶

```python
# torch/_inductor/fx_passes/post_grad.py:366-375
if config.bucket_all_reduces_fx != "none":
    bucket_all_reduce(gm, bucket_size_determinator, mode)
```

两个 all-reduce 输入拼接后只执行一次 collective，等待完成后再拆回原 shape；无关 `mm` 保留在完整社区函数里。GPU 单 rank reference 与 NPU 双 rank HCCL 均通过，NPU collective 2→1。Event p50/p99 改善 21.41%/17.69%，但 reserved memory 从约 6 MiB 增到 24 MiB；且只测一个消息规模，因此保持上游默认 `none`，作为同 shape 模型级候选，而不是直接全局开启。

### 5.3 reduce-scatter 分桶

```python
# torch/_inductor/fx_passes/post_grad.py:348-363
if config.bucket_reduce_scatters_fx != "none":
    p = bucket_reduce_scatter
    p(gm, bucket_size_determinator, bucket_mode)
```

两个 bf16 输入被拼接后执行一次 reduce-scatter，再切回两个输出。GPU 单 rank功能 reference 通过；NPU 双 rank HCCL 数值和 2→1 改写通过。Event p50 改善 6.05%，但 p99 回退 10.32%，峰值 allocated 增加约 0.56 MiB，因此判 `PERF_MIXED` 并保持默认 `none`。

## 6. 真实问题、最小适配和修复

### 6.1 FX 观察器读取了缓存代码

问题：`GraphModule.code` 可能在 pass 未调用 `recompile()` 时仍是旧缓存，导致常量折叠 before/after 文本相同，而 compile-debug 的 post-pass 图已经变化。

```python
# runners/native_fx_observer.py
def current_graph_code(gm):
    return gm.graph.python_code(root_module="self").src
```

修复只改变 tracker 的只读渲染，不调用 `recompile()`，不改变社区测试和编译生命周期。

### 6.2 T-082 扫错 PatternMatcherPass 容器

问题：`pointless_convert` 注册在 `patterns`，而 view/permute pair 注册在 `early_patterns`。首轮 worker 只扫描前者，在导入 torch 后、真正编译前报“未找到精确目标注册”。

```python
# runners/t081_t083_performance_worker.py
for registry in (joint_graph.early_patterns, joint_graph.patterns):
    for entries in registry.patterns.values():
        ...
```

修复后仍只包装精确 handler，不关闭整个 pattern matcher。

### 6.3 环境脚本非致命返回值触发 `set -e`

`activate_pass.sh` 最后清理可选 alias；别名不存在时某些 shell 返回 1，首轮一键脚本在导入 torch 前静默退出。启动器现在临时关闭 `set -e` 完成 source，再以 `CONDA_DEFAULT_ENV=Pass` 和 Python 绝对路径强校验，既不掩盖激活失败，也不误杀已成功激活的环境。

### 6.4 NPU 设备启动瞬时失败

T-082 view 的一次串行重测在 OFF3 遇到 `507033/E39007`：HDC/TSD 设备子进程启动超时。它发生在 `runtime.set_device()`，尚未进入 pass/编译。该轮按设计拒绝聚合；换空闲 NPU 4 从 OFF1 完整重跑后通过。没有用代码绕过设备错误。

### 6.5 并发预跑污染风险

不同物理 NPU 仍共享 CPU 编译、主机调度和 HCCL 资源。并行预跑结果未纳入正式汇总；正式入口新增全局 `flock`，后续 tracker 性能任务并发时直接拒绝。除 constant-fold/all-gather 原本独占外，其余单元均重新串行测量。

## 7. 证据导航

- GPU复核：[t081_t083_gpu_reference_review_20260908.md](t081_t083_gpu_reference_review_20260908.md)
- NPU功能紧凑汇总：`results/current/T-081～T-083/npu_functional_summary.json`
- 逐单元功能原件：`results/current/T-081～T-083/functional/*.json`
- 逐单元性能门禁：`results/current/T-081～T-083/performance_gates/*.json`
- 正式性能结论：`results/current/T-081～T-083/performance_summary.json`
- 全局矩阵：[current_acceptance_unit_matrix.md](current_acceptance_unit_matrix.md)
- 一键功能入口：`scripts/run_t081_t083_npu_function.sh`
- 一键性能入口：`scripts/run_t081_t083_performance.sh` 或统一入口 `scripts/run_npu_performance_task.sh`

原始运行目录位于 `/home/z50063656/tmp/t081-npu-functional-results/`、`t082-*`、`t083-*`；仓库记录保存原件绝对路径和 SHA-256。原始 debug/cache 不提交，避免仓库膨胀。
