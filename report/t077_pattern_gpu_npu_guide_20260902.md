# T-077 pattern 源码、意图与 GPU/NPU 行为对照

> 更新时间：2026-09-02 22:54 CST（UTC+08:00）
> PyTorch：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> GPU reference：A100，11/11 direct cases valid，17/17 variants valid
> NPU：Ascend910B2，`triton_experimental`；5/5 单元正式闭环；MM 发现 1 个 lowering 回归并已验证本地修复

本文是 `results/current/AU-apply-gumbel-max-trick/`、`AU-b2b-gemm/` 和三个
`AU-decompose-mem-bound-mm-*` 目录中结构化结果的学习视图。每个 comparison JSON 也保存
`intent/source_locations/gpu_behavior/npu_behavior`，便于脚本消费和逐项复核。

## 1. Gumbel-max 等价采样重写

原图先做 softmax，再除以参数为 1 的指数分布随机数，最后 argmax。利用 Gumbel-max 等价关系，
pass 删除 softmax 和除法，改成 `logits + (-log(rand_exp))` 后 argmax，减少无必要的归一化。

```python
# torch/_inductor/fx_passes/apply_gumbel_max_trick.py:10-68
@register_graph_pattern(
    CallFunction(torch.argmax,
        CallFunction(operator.truediv, KeywordArg("softmax"), KeywordArg("rand_exp")),
        dim=-1, keepdim=True),
    pass_dict=apply_gumbel_max_trick_pass,
)
def apply_gumbel_max_trick(match, softmax, rand_exp):
    if rand_exp.target != "exponential_" or rand_exp.args[1] != 1.0:
        return
    ...
    log = graph.call_function(torch.log, (rand_exp,))
    gumbel_noise = graph.call_function(operator.neg, (log,))
    argmax_input = graph.call_function(operator.add, (logits, gumbel_noise))
    counters["inductor"]["apply_gumbel_max_trick"] += 1
```

| variant | GPU 行为 | NPU 行为 | 结论 |
| --- | --- | --- | --- |
| `distribution-positive` | 命中 1 次；`1000000x10` 采样频率满足 10% 相对容差 | 命中 1 次；实际/期望比 `0.9913~1.0044` | `BEHAVIOR_UNCHANGED` |

该 handler 没有设备白名单，因此 NPU 能直接复用上游 rewrite；这里验证的是统计分布合同，不是逐元素
随机结果相等。

## 2. B2B GEMM

B2B GEMM 寻找 `(A@B)@C` 或 `A@(B@C)`，中间允许只有单输入 pointwise 子图。收益判断比较候选
tile 的平均 load ratio，并保留未优化 choice 参与 autotune。当前设备门禁只接受 CUDA/XPU。

```python
# torch/_inductor/fx_passes/b2b_gemm.py:361-429
def is_b2b_gemm_good_on(is_left_assoc, A_node, B_node, C_node):
    fake_tensors = A_node.meta["val"], B_node.meta["val"], C_node.meta["val"]
    if not check_all_attr_true(fake_tensors, "is_cuda") and not check_all_attr_true(
        fake_tensors, "is_xpu"
    ):
        return False
    ...
    return average_ratio > 1
```

```python
# torch/_inductor/fx_passes/b2b_gemm.py:599-639
@register_graph_pattern(CallFunction(torch.ops.aten.mm, Arg(), Arg()), pass_dict=B2B_GEMM_PASS)
def b2b_gemm_handler(match, mat1, mat2):
    # 从 inner mm 沿唯一 pointwise 用户链寻找 outer mm
    ...
    if not outer_mm:
        return
```

| variant | 意图 | GPU 行为 | NPU 行为 | 结论 |
| --- | --- | --- | --- | --- |
| `left-gelu-positive` | 左结合两级 mm，中间 GELU | 命中 B2B，fp16/fp32 accumulation 合同通过 | device guard 拒绝；原始 mm+GELU+mm 正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `right-relu-positive` | 右结合两级 mm，中间 ReLU | 命中 B2B | guard 拒绝；原图正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `trivial-left-positive` | 左结合纯两级 mm | 命中 B2B | 保留两个 extern mm | `EXPECTED_PRODUCT_DIVERGENCE` |
| `trivial-right-positive` | 右结合纯两级 mm | 命中 B2B | 保留两个 extern mm | `EXPECTED_PRODUCT_DIVERGENCE` |
| `bad-pattern-negative` | inner mm 存在非唯一数据流时不得吞并 | 不命中 | 不命中，原图正确 | `BEHAVIOR_UNCHANGED` |
| `bad-shape-negative` | 100x100 非盈利 shape 不选 B2B | 不命中 | 不命中，原图正确 | `BEHAVIOR_UNCHANGED` |

正例差异不能仅靠“数值正确”掩盖：NPU 的目标 counter 和左右 entrance marker 都是 0，首个分歧明确位于
device guard。要支持 NPU 不只是把 `is_npu` 加进条件，还要证明 Ascend 模板及 autotune 收益。

## 3. memory-bound BMM decomposition

该 pattern 针对“大 batch、小 M/K/N”的 BMM。分解后不再调用 bmm，而是显式 broadcast multiply 后沿
K 求和；这样在特定 memory-bound 形状上可能比通用 GEMM 更合适。

```python
# torch/_inductor/fx_passes/decompose_mem_bound_mm.py:61-90,224-240
def should_decompose_bmm(mat1, mat2):
    if check_device(mat1, mat2, device="cuda") or check_device(mat1, mat2, device="xpu"):
        if mat1.shape[0] < 10240:
            return False
        if sum(dim < 32 for dim in (M, K, N)) < 2:
            return False
        return True
    ...

def decompose_bmm(match, mat1, mat2):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2).to(mat1.dtype)
```

| variant | GPU 行为 | NPU 行为 | 结论 |
| --- | --- | --- | --- |
| `large-batch-small-mnk-positive` | `B=10240,M=K=N=2` 的 forward/backward 均分解 | 干净 NPU 0 上 counter=0，保留 bmm；forward 与两路梯度正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `other-dims-threshold-negative` | `K=N=32` 不满足“小于 32”，不分解 | 同样不分解；forward/backward 正确 | `BEHAVIOR_UNCHANGED` |
| `batch-threshold-negative` | `B=2000` 小于 10240，不分解 | 同样不分解；forward/backward 正确 | `BEHAVIOR_UNCHANGED` |

NPU 的 3/3 合同是在执行前显示 `No running processes found in NPU 0` 的卡上完成，因而此前占用卡上的
临时结果已被替代。正例的差异发生在 `should_decompose_bmm` device guard，而不是 lowering 或数值层。

## 4. memory-bound MM decomposition

MM 使用相同的 multiply+sum 思路，但收益条件改为“大 M、小 K/N”。静态分支要求 M 至少 10240，且
K/N 严格小于 32；dynamic 分支用 symbolic-shape 判断放宽“目前无法证明”的动态维。

```python
# torch/_inductor/fx_passes/decompose_mem_bound_mm.py:93-209,267-285
def should_decompose_mm(mat1, mat2):
    ...
    return (
        (check_device(mat1, mat2, device="cuda") or check_device(mat1, mat2, device="xpu"))
        and statically_known_true(mat1.shape[0] >= 10240)
        and statically_known_true(mat2.shape[0] < 32)
        and statically_known_true(mat2.shape[1] < 32)
    ) or cpu_contract

def decompose_mm(match, mat1, mat2):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2).to(mat1.dtype)
```

| variant | GPU 行为 | NPU 行为 | 结论 |
| --- | --- | --- | --- |
| `fp32-large-m-small-kn-positive` | `20480x5 @ 5x2` 的 forward/backward 分解 | device guard 拒绝，改走 extern mm；修复候选下前向/两路梯度误差均为 0 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `fp32-k-threshold-negative` | `K=32` 不分解 | 保持 extern mm；前向/两路梯度误差均为 0 | `BEHAVIOR_UNCHANGED` |
| `fp32-m-threshold-negative` | `M=2048` 不分解，forward/backward 正确 | 目标 pass 不命中，但安装态后续 small-mm lowering 使左梯度 4085/4096 不一致；修复后误差为 0 | `NPU_REGRESSION`（修复已验证） |
| `mixed-large-m-small-kn-positive` | bf16 autocast 下仍分解且梯度正确 | guard 拒绝；cast 走 Triton、mm 走 extern，修复候选下正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `mixed-k-threshold-negative` | bf16 autocast、K=32 不分解 | 同样不命中并正确 | `BEHAVIOR_UNCHANGED` |
| `mixed-m-threshold-negative` | bf16 autocast、M=2048 不分解 | 与 fp32 tiny-mm 共用保护；修复候选下正确 | `NPU_REGRESSION` 防回归覆盖 |

上游测试的 `has_bias=True/False` 参数没有参与 mm 图，T-077 运行仍覆盖了 12 个生成测试实例，但不会把
它机械计成两倍 optimization contract。

这里发现的回归并不在 `decompose_mm` 本身。捕获的 transformed FX 仍是 `aten.mm`，随后 upstream
lowering 还会独立判断是否把小 K/N 的 mm 展开成 pointwise：

```python
# torch/_inductor/kernel/mm.py:394-398
if out_dtype is None and _use_small_mm_pointwise(m, k, n, layout):
    counters["inductor"]["decompose_mm_pointwise"] += 1
    mat1 = L.unsqueeze(mat1, -1)
    mat2 = L.unsqueeze(mat2, 0)
    return L.sum_(L.mul(mat1, mat2), axis=1)

# torch/_inductor/kernel/mm_common.py:197（heuristic 定义入口）
# 该启发式面向 CUDA/XPU，但在 NPU experimental handler 复用 upstream mm lowering 时也会被调用。
```

原安装态的 `fp32-m-threshold-negative` 前向最大误差只有 `4.768e-7`、右梯度误差为 0，但左梯度
最大绝对误差为 `29.026336669921875`，`4085/4096` 个元素不一致；生成代码中反向 `mm_2` 被拆成
`triton_unk_fused_mm_0/1`。这把首个分歧明确定位到 lowering，而不是 pattern matching。

修复候选只在 NPU 上关闭该 heuristic，并保持其他设备原行为：

```python
# torch_npu/_inductor/triton_experimental/overrides.py
def _disable_small_mm_pointwise_on_npu():
    current = mm_kernel._use_small_mm_pointwise
    if getattr(current, "_torch_npu_small_mm_guard", False):
        return

    def npu_safe_small_mm_pointwise(m, k, n, layout):
        if layout.device.type == "npu":
            return False
        return current(m, k, n, layout)

    npu_safe_small_mm_pointwise._torch_npu_small_mm_guard = True
    mm_kernel._use_small_mm_pointwise = npu_safe_small_mm_pointwise
```

候选提交为 `dfbcc25b76743ea6c1c5cd61b6b30f0a910148a6`（本地 detached，尚未推送/合入）。两个定向单测
2/2 通过，且同一 MM 六变体 6/6 的 forward、left-grad、right-grad 误差全部为 0。补丁副本位于
`issues/REF-decompose-mm-native/backend_fix_dfbcc25.patch`。

## 5. dynamic addmm decomposition

addmm 复用 `should_decompose_mm(mat1, mat2)`，replacement 在 multiply+sum 结果上加 bias。动态测例通过
`skip_dynamic_shape_dim_check=True` 验证超大动态 M，而不把 M 缩小为更容易运行的替代形状。

```python
# torch/_inductor/fx_passes/decompose_mem_bound_mm.py:243-264
@register_graph_pattern(CallFunction(aten.addmm, Arg(), Arg(), Arg()), ...)
def decompose_addmm(match, bias, mat1, mat2):
    def repl(bias, mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2).to(mat1.dtype) + bias
    if should_decompose_mm(mat1, mat2):
        counters["inductor"]["decompose_addmm"] += 1
        match.replace_by_example(repl, [bias, mat1, mat2])
```

| variant | GPU 行为 | NPU 行为 | 结论 |
| --- | --- | --- | --- |
| `dynamic-large-m-small-kn-positive` | 原始 `M=19494144,K=N=8`、dynamic=True 命中 1 次 | 同一原始尺寸、dynamic=True 下 counter=0，保留 `extern_kernels.addmm` 且数值正确 | `EXPECTED_PRODUCT_DIVERGENCE` |

这一项没有缩小 `M`，也没有关闭 dynamic。GPU 证明“该 optimization contract 应命中”，NPU 则证明
“当前产品因 device guard 明确不命中，但 fallback 路径能够承载原始超大 workload”。

## 6. 阅读结论的方法

- `BEHAVIOR_UNCHANGED` 表示同一正向 rewrite 或负向 guard 在两端合同一致。
- `EXPECTED_PRODUCT_DIVERGENCE` 表示有明确源码设备边界；它不是 correctness 失败，也不能自动变成
  “应删除 guard”的修复任务。
- counter/marker 回答“目标 pattern 是否发生”，数值与梯度回答“实际路径是否正确”，两者不能互相替代。
- T-077 五个单元均已正式闭环：Gumbel-max 命中一致；B2B、BMM、MM 正例和 dynamic addmm 在
  NPU device guard 处分叉；相关负例通常保持一致。
- MM 的 `M=2048,K=N=2` 负例额外暴露了与目标 pattern 独立的 NPU lowering correctness 回归；
  修复已用同一 community contract 验证，但候选尚未推送/合入，因此仍保留在 known issues。
