# T-086 功能与性能测例讲解

> 更新时间：2026-09-09 00:43:01 CST（UTC+08:00）
>
> 当前状态：零设备准备完成；1 个 acceptance unit、2 个原生 GPU cases、2 个 variants 等待实际设备运行。本文不记录任何 GPU/NPU 通过结论。

T-086 对 backlog 的 5 个 provisional units 做了源码回查。只有
`AU-post-grad-reinplace-inplaceable-ops` 具备实际 GPU 原生正负合同，进入本批执行分母；其余 4 项
保留为 deferred。冻结 PyTorch commit 为
`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。

NPU 动态验证、修复复核和性能测量只能使用 `triton_experimental`。后端必须在导入
`torch`/`torch_npu` 之前选择，OFF/ON 必须是不同进程。显式产品关闭时记录关闭证据并免测，不得绕过。

## GPU 一键执行

GPU 机器拉取包含 T-086 的提交后，从数据盘临时目录启动：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-086 \
  --gpu 2 \
  --wait-gpu
```

排障时也可在已激活 `PassGPURef` 且环境变量已配置的终端使用任务专属入口；该入口不会自动等卡或
生成文本 handoff，因此日常执行仍以上述共享入口为准：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t086_reference_all.sh \
  --pytorch-root /data/z50063656/src/pytorch \
  --output-root /data/z50063656/tmp/t086-reference-results
```

两条原生方法都必须实际创建 CUDA tensor、完成编译和执行；CPU、FakeTensor、skip、xfail、只解析图均
不能替代本轮 GPU reference。成功后 handoff 由共享入口写到：

```text
/data/z50063656/tmp/t086-reference-results/latest-text-handoff.json
```

网页单文件超限时上传同一轮 `latest/text-handoff-parts/` 内的 `manifest.json` 和全部
`part-*.json` 到 `results/incoming/T-086/`，不能混入其他运行。

## pass 所在位置与执行阶段

`reinplace_inplaceable_ops` 位于 post-grad 末段。在图已经 functionalize 后，它判断 functional
mutation 是否能安全恢复为原位写；随后才进入剩余分解、lowering、scheduler/codegen 和可能的 autotune。

```python
# torch/_inductor/fx_passes/post_grad.py:462-466
# Keep these last, since they introduce mutation.
GraphTransformObserver(gm, "reinplace_inplaceable_ops").apply_graph_pass(
    functools.partial(reinplace_inplaceable_ops, fake_tensor_updater),
)
```

目标 pass 先规范化 scatter，更新 fake tensor alias 信息，再执行核心重入原位判断，最后还原 scatter：

```python
# torch/_inductor/fx_passes/reinplace.py:1062-1073
def reinplace_inplaceable_ops(fake_tensor_updater, graph):
    with enable_python_dispatcher():
        canonicalize_view_scatter_ops(graph)
        fake_tensor_updater.incremental_update()
        reinplace_inplaceable_ops_core(graph)
        decompose_generalized_scatter(graph)
```

本批关注 `aten.index_put.default → aten.index_put_.default`。核心意图不是“任何 index_put 都原位化”，
而是在输入不再存活、view/alias/copy-back关系安全时消除额外输出与 copy；只要原输入仍被后续观察，就必须
保留 functional 路径。

## 功能测例

### 正例：dead-input index_put 可以重入原位

社区位置：
`test/inductor/test_torchinductor.py::GPUTests.test_index_put_reinplace_cuda`。

```python
# test/inductor/test_torchinductor.py:11301-11311
def fn(x, idx):
    src = torch.ones(idx.size(0), device=x.device)
    x.index_put_((idx,), src)
    return x.expand((2, x.shape[0]))

a = torch.randn(1024)
idx = torch.arange(10)
self.common(fn, (a, idx))
assertGeneratedKernelCountEqual(self, 1)
```

源码里的 fixture 最初在 CPU 构造，但方法被 `copy_tests(CommonTemplate, GPUTests, GPU_TYPE)` 物化为
CUDA 方法；`GPUTests.common=check_model_gpu` 会保持 stride 地复制输入到 `GPU_TYPE`，所以运行生成的
`..._cuda` 方法属于实际设备测试，不是 CPU 测试。

验证意图：compiled/eager输出相等，输入 mutation 相等，返回 expand 的 stride/alias不变；post-grad
中 functional `index_put` 被安全改成 `index_put_`，社区以一个生成 kernel 约束收益结构。

### 负例：live input 必须阻止重入原位

社区位置：
`test/inductor/test_torchinductor.py::GPUTests.test_index_put_failed_reinplace_cuda`。

```python
# test/inductor/test_torchinductor.py:11313-11323
def fn(x, idx):
    src = torch.ones(idx.size(0), device=x.device)
    y = x.index_put((idx,), src)
    return x, y

a = torch.randn(1024)
idx = torch.arange(10)
self.common(fn, (a, idx))
assertGeneratedKernelCountEqual(self, 2)
```

返回值同时暴露原输入与更新结果。如果错误改为 `index_put_`，原输入会被污染。该负例要求两个输出均与
eager一致，并以两个生成 kernel 固定“不得重入原位”的结构边界。

### GPU/NPU 待验证对照

GPU 先运行以上两个原生方法，采集 `fx_graph_readable.py`、`fx_graph_transformed.py`、`output_code.py`
及 IR。GPU 通过后才进入 NPU。NPU 必须在 `triton_experimental` 下复用相同 shape、index数量、live/dead
关系，分别确认：

- 正例：发生精确 `index_put → index_put_` 改写，数值、输入 mutation、stride 和 alias 正确；
- 负例：保留 functional `index_put`，原输入未被污染；
- 两例：`fullgraph=True`，无 graph break/CPU fallback。

当前只是准备计划，尚未得到上述实际设备结果，因此不能写成“GPU/NPU行为一致”。

这里的 fallback 需按层级解释。当前控制节点 `triton_experimental` 已为 `index_put/index_put_` 注册设备
extern lowering；它不是 Dynamo graph break 或回退到 CPU：

```python
# torch_npu/_inductor/triton_experimental/lowering.py:476-491
def npu_index_put(x, indices, values, accumulate=False):
    x = _index_put_clone(x)
    values = to_dtype(values, x.get_dtype())
    return _index_put_fallback(x, indices, values, accumulate)

def npu_index_put_(self, indices, values, accumulate=False):
    values = to_dtype(values, self.get_dtype())
    return _index_put_fallback(self, indices, values, accumulate)
```

因此 ON 路径的理论收益是从 out-of-place `clone + IndexPutFallback` 变成直接对 self 的
`IndexPutFallback`，而不是把 extern 算子变成 Triton scatter kernel。该源码只是准备期静态发现；正式
结论必须绑定实际运行加载文件的 sha256、生成代码和设备结果。门禁字段 `fallbacks=0` 只表示没有
graph break/CPU fallback，注册的 NPU `IndexPutFallback` 会单独记录为允许的设备 lowering。

## 性能测例

### 来源审计

在冻结 commit 的 `benchmarks/`、`test/inductor/` 和 `test/distributed/` 搜索 `reinplace`、
`index_put`、`inplaceable_ops`、`generated_kernel_count`，没有发现可精确切换同后端目标 pass 的社区
benchmark。社区现有证据是功能、mutation/alias 与 kernel-count 测试。

因此性能 workload 是从上述社区正例派生，而不是社区原生 benchmark：保持 `x=[1024] fp32`、
`idx=arange(10) int64`、`src=ones(10)`、in-place `index_put_` 和 expand 返回完整不变。负例不参与计时，
但必须在功能门禁中执行。

### OFF/ON 控制与测量边界

OFF 仅将 `post_grad` 模块内的 `reinplace_inplaceable_ops` 替换为同签名观察空操作；ON 调用原实现。
不会关闭整个 post-grad、pattern matcher 或其他 pass。ON 必须在 pass 的 before/after 图中观察到
`index_put.default` 减少且 `index_put_.default` 增加；OFF 图不得因目标 pass 改变。

测量的是一次 compiled 社区目标函数调用，包含 index_put 和 expand 返回。它是目标级子图，明确属于
非端到端 benchmark：不包含模型前后处理、训练 step、优化器、数据加载，结论不得推广为模型吞吐收益。

正式测量顺序固定为：

```text
OFF1 → ON1 → ON2 → OFF2 → OFF3 → ON3
```

每臂独立进程，10 次预热、100 个样本；记录 host 同步 wall time、设备 Event p50/p99、编译耗时、
allocated/reserved峰值、全部原始样本、PID、backend、loaded source sha256、FX、IR 与 generated code。

### 实际设备功能门禁与性能命令

先在目标设备生成独立 OFF/ON 功能原件：

```bash
cd /home/z50063656/tmp

export PASS_TRACKER_WORK_DIR=/home/z50063656/tmp
export ASCEND_RT_VISIBLE_DEVICES=0
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t086_performance.py \
  --unit reinplace-index-put \
  --device npu \
  --phase functional
```

人工复核 GPU reference、上述功能原件、产品是否明确关闭以及源码 hash 后，生成签名 gate；再执行：

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t086_performance.py \
  --unit reinplace-index-put \
  --device npu \
  --phase benchmark \
  --gate /path/to/reviewed-gate.json
```

worker 不提供绕过产品 disable 的选项；门禁若记录 `product_disabled=true` 会直接拒绝性能运行。
NPU 的注册 `IndexPutFallback` extern lowering 允许存在，但必须保留生成代码；任何 graph break 或 CPU
fallback 仍会使门禁失败。

## Deferred 候选

- `AU-post-grad-reciprocal-sqrt-to-rsqrt`：社区测试直接断言 rsqrt/codegen/counter，但
  `x=torch.rand(64)+1.0` 没有 device，实际只编译 CPU。需要先保留原生事实，再审核只增加 device 的
  最小适配；当前不进入 GPU 分母。
- `AU-post-grad-remove-assert-ops`：没有直接 post-grad 社区测试；profiler测试不能替代。
- `AU-post-grad-remove-noop-ops`：当前 post-grad候选仅覆盖“不得删除”的分布式负例；可删除正例属于
  pre-grad另一条 pass，不能跨后端、跨 pass 借用。
- `AU-post-grad-remove-profiler-ops`：直接社区测试输入为 CPU，尚无实际 GPU 合同。

Deferred 项不执行、不计数、不做性能测量。获得直接实际设备合同或按工作流审核的最小适配后，再分配
后续 T 编号重开，不为填满每批数量而降级证据。

## 零设备校验

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_t086.py

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/tests/test_t086_preparation.py
```

这两条命令只解析源码和合同，不导入 torch，也不访问 GPU/NPU；通过仅表示“可交给 GPU 运行”，不表示
任何设备测试成功。
