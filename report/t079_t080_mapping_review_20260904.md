# T-079/T-080 acceptance-unit 映射审核

> 更新时间：2026-09-04 08:55 CST（UTC+08:00）
> PyTorch：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> 状态：7 个单元、17 个 direct cases、27 个动态 variants 已准备，等待 GPU reference

## 1. 批次边界

| 任务 | Acceptance units | Cases | Variants | 选择理由 |
| --- | ---: | ---: | ---: | --- |
| T-079 | 4 | 4 | 14 | 每个社区方法同时覆盖正例与关键 guard，均在 GPU 上真实编译 |
| T-080 | 3 | 13 | 13 | 补正两个 no-test-found 映射，并优先保留社区端到端/性能设计 |

T-079/T-080 只完成 mapping、GPU reference plan 和一键入口，不提前形成 NPU verdict。GPU suite
全部 `reference_valid=true` 后才能冻结分母；随后 NPU 必须在 fresh process 中先选择
`triton_experimental`，再导入 `torch/torch_npu`。

## 2. T-079：矩阵与 cat/split 消除

### 2.1 batch=1 bmm → mm

位置：`torch/_inductor/fx_passes/joint_graph.py:979`。

```python
def bmm_to_mm(match, mat1, mat2):
    def repl(a, b):
        return torch.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)

    if check_device(...) and mat1.meta["val"].shape[0] == 1:
        match.replace_by_example(repl, [mat1, mat2])
```

意图是在 batch 唯一时去掉批处理维，走普通 mm；社区方法同时断言 batch=1 生成 `extern mm`，
batch=3 保留 `extern bmm`。NPU 侧的 GPU-oriented device helper 先按 capability pending 处理，不能
直接解释为产品显式关闭。

### 2.2 cat→slice→cat 折叠

位置：`torch/_inductor/fx_passes/post_grad.py:1089`。

```python
def cat_slice_cat(match, cat_input, size, dim=1):
    first, *rest = cat_input
    if size >= 0 and statically_known_leq(size, first.get_size()[dim]):
        return L[aten.cat]([first, *rest, L[aten.slice](first, dim, 0, size)], dim)
    return original_two_cat_fallback(...)
```

这里必须区分“pattern 结构命中”和“折叠真正生效”：越界 end 或负 end 仍可进入 handler，但只能返回
等价 fallback，不能记成 `REWRITE_APPLIED`。

### 2.3 split_with_sizes→cat 消除

位置：`torch/_inductor/fx_passes/post_grad.py:1731`。

```python
def splitwithsizes_cat_replace(match, input_):
    return input_
```

只有全部 getitem 按原顺序、同一维度重新拼回时才能直接返回输入；缺片、异维或重排都会改变语义，
社区测试明确要求 counter 为 0。

### 2.4 cat→split_with_sizes 消除

位置：`torch/_inductor/fx_passes/post_grad.py:1786`。

```python
def cat_splitwithsizes_replace(match, input_):
    return input_
```

该方向还要求 cat 没有额外用户，split 数量及每段边界与原 cat 输入完全一致。它与上一方向的 guard
不同，因此保留为独立 acceptance unit。

## 3. T-080：访存、softmax 与构造器移动

### 3.1 full+scatter → pointwise where

位置：`torch/_inductor/fx_passes/joint_graph.py:1150-1225`。

```python
def scatter_upon_const_tensor_extra_check(match):
    ...
    return selector_ft.size(dim) == 1

def scatter_upon_const_tensor(match, shape, background_val, dtype, dim, selector, val):
    metrics.num_matches_for_scatter_upon_const_tensor += 1
    # 构造 broadcast index，再用 where 选择 val/background_val
```

意图是避免先物化完整常量张量再做 mutation scatter。T-074 将其标成 no-test-found，但社区已有
专属 `test_scatter_optimization.py`，包含正例、密度/shape/非 const 负例、低精度回归，以及
CrossEntropy backward 端到端测例。该 E2E 方法在 `DO_PERF_TEST=1` 时还提供 A100 原始 shape、
设备计时与峰值内存设计，后续性能阶段优先复用。

### 3.2 prepare_softmax → online primitive

位置：`torch/_inductor/fx_passes/post_grad.py:490-514`。

```python
def prepare_softmax_pattern(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsub = x - xmax
    xexp = xsub.exp()
    return xmax, xexp.sum(dim=dim, keepdim=True), xsub, xexp

def prepare_softmax_replacement(x, dim):
    xmax, xsum = prepare_softmax_online(x, dim)
    ...
```

目的是让 max 与 sum 使用 online-softmax primitive，减少遍历和中间量。上游 guard 只列
`cuda/xpu` 且要求对应 backend 为 Triton；这不是 NPU 产品显式 disable。NPU 需要先记录原生 gate，
再单独评审最小适配，功能命中前不得跑性能。社区已有两套 `DO_PERF_TEST` 设计可供后续复用。

### 3.3 CPU constructor 安全迁移到加速器

位置：`torch/_inductor/fx_passes/post_grad.py:2488`。

```python
def move_constructors_to_gpu(graph):
    ConstructorMoverPass(get_gpu_type(), allow_inputs=..., allow_outputs=...)(graph)
```

可移动的 `arange(..., device="cpu").to(accelerator)` 应避免独立 copy kernel；但为 `index_put_`
提供标量的 constructor 存在受保护依赖，必须留在 CPU。NPU 不能靠 CUDA token 缺失判 PASS，需建立
`triton_experimental` 等价的 kernel/copy artifact 断言。

## 4. copy_tests 入口纠偏

PyTorch `copy_tests(CommonTemplate, GPUTests, GPU_TYPE)` 的真实实现会给方法追加 `_cuda`。旧 T-078
静态展开错误地接受了不存在的无后缀方法。本轮已修正 runner/validator，并将两个 addcdiv nodeid
纠正为：

```text
GPUTests.test_addcdiv_fma_bitwise_equal_cuda
GPUTests.test_addcdiv_fma_uses_fma_and_div_rn_cuda
```

修正后 T-078、T-079、T-080 的零设备静态校验全部通过；尚未在本控制节点运行 GPU。
