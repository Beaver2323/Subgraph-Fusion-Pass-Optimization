# T-081～T-083 GPU reference复核报告

> 复核时间：2026-09-08T00:33:56+08:00
> 结论：11/11 cases、24/24 variants有效，新增7个acceptance units的GPU分母已冻结；NPU功能、GPU/NPU对照和性能仍未执行。
> NPU硬约束：动态验证与性能只使用`triton_experimental`，每个OFF/ON及跨backend分支使用fresh process。

## 1. 输入、环境与冻结结论

三个handoff均通过整包/分片SHA256、原文恢复、suite完整性和计划映射校验。它们使用同一环境指纹
`ad8df1e2357e5e0351014dc7c3dd578f8ae640b4726d1138bc991cc30da10dc3`：PyTorch
`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` clean tree、A100-SXM4-80GB(sm80)、CUDA
12.6、Triton 3.8.0；物理节点8卡，本轮只暴露`CUDA_VISIBLE_DEVICES=2`。11个case均直接运行原生社区
方法，未启用adapter，skip/xfail/缺例均为0。

| 任务 | run | cases | tests | variants | 数值证据case | 仅结构case | 冻结单元 |
|---|---|---:|---:|---:|---:|---:|---:|
| T-081 | `reference-20260907T151850+0800-lypgqs3_` | 3/3 | 7 | 10/10 | 2 | 1 | 2 |
| T-082 | `reference-20260907T162213+0800-l78yijaq` | 4/4 | 4 | 8/8 | 1 | 3 | 2 |
| T-083 | `reference-20260907T162312+0800-8lgd6yen` | 4/4 | 5 | 6/6 | 2（仅单rank） | 2 | 3 |

“冻结”只表示社区合同、case和variant分母不再漂移，不表示NPU已经通过。结构case仍必须先补同输入数值
门禁，才可进入性能。

## 2. T-081：constant fold与pointless convert

### 2.1 `constant_fold_uniform_value`

源码调用链：

```text
# torch/_inductor/compile_fx.py::_recursive_joint_graph_passes
torch.compile
  -> AOTAutograd joint graph
  -> torch/_inductor/fx_passes/joint_graph.py:699 joint_graph_passes
  -> torch/_inductor/fx_passes/joint_graph.py:728 GraphTransformObserver(...)
  -> torch/_inductor/fx_passes/joint_graph.py:501 constant_fold_uniform_value
  -> UniformValueConstantFolder.run
  -> replace uniform tensor with aten.full
  -> remove_no_ops / remove_redundant_views
  -> post_grad / lowering / scheduling / codegen
```

核心代码：

```python
# torch/_inductor/fx_passes/joint_graph.py:501-599
def constant_fold_uniform_value(gm):
    cf = UniformValueConstantFolder(gm)
    cf.run()
    # 将可安全重建的均匀常量替换成full；自指shape会被拒绝
    if _has_self_referential_shape(shapes, node):
        continue
    new_node = graph.call_function(aten.full.default, args=(shapes, value), ...)
    node.replace_all_uses_with(new_node)
    remove_no_ops(gm, zeros, ones)
    remove_redundant_views(gm)
```

意图是把`full(1)-1`折成零、把`full(-1)+2`折成一，再删除`x+0`/`x*1`；同时避免新full的
shape依赖被替换节点而形成环。GPU原生dynamic与self-referential两个case均通过eager/compiled正确性。
编译后FX把：

```python
# GPU before：原生测试输入图
full = aten.full([s0, s1], 1)
sub = aten.sub(full, 1)
add = aten.add(x, sub)
return add

# GPU post-pass：compile-debug可见
full_default = aten.full([s0, s1], 0.0)  # 已无用户，后续可删
return x
```

本轮发现观察器显示缺口：`constant_fold_uniform_value`直接改`gm.graph`但不调用`gm.recompile()`，旧runner
随后读取缓存的`gm.code`，所以它的入口before/after文本相同；同一run的compile-debug post-pass图和社区
数值断言仍证明变换生效。最小修复只改tracker观察器为
`gm.graph.python_code(root_module="self").src`，不改测试体、设备、断言或被测PyTorch。

GPU结论：4个variant有效。NPU结论：尚无；需在`triton_experimental`下验证动态shape、实际恒等改图、
数值与无fallback。性能：未解锁。

### 2.2 `pointless_convert`

```python
# torch/_inductor/fx_passes/joint_graph.py:842
@patterns.register(...)
def pointless_convert(match, arg, dtype1, dtype2):
    # 只有往返转换不会损失精度/舍入语义时，才移除中间转换链
    ...
```

GPU运行了五个社区参数化方法，覆盖fp32↔fp16/bf16/fp64、emulated narrowing与整数往返保护，结构计数
符合社区断言，6个variant有效。但这些方法是`make_fx -> joint_graph_passes -> 节点计数`，没有执行编译
设备kernel，也没有输出数值oracle。因此GPU只冻结结构合同，不能直接声称NPU正确或有性能收益。

## 3. T-082：互逆permute/view消除

### 3.1 `pointless_permute_pair`

```python
# torch/_inductor/fx_passes/joint_graph.py:957
def pointless_permute_pair(match, arg, perm1, perm2):
    # 两次permute互逆且中间结果没有额外用户时，用原输入替换输出
    ...
```

GPU 2D与3D正例都从两个`aten.permute`变为直接`return x`；中间permute同时作为输出的负例保留两个
节点：

```python
# GPU positive before/after
y = aten.permute(x, [1, 0])
z = aten.permute(y, [1, 0])
return z
# -> return x

# GPU intermediate-user negative：保持不变
return (y, z)
```

这是精确结构证据，不含数值执行。NPU必须另验shape、stride、alias/storage关系和数值。

### 3.2 `pointless_view_pair`

```python
# torch/_inductor/fx_passes/joint_graph.py:936-946
def pointless_view_pair(match, arg, size1, size2):
    if guard_size_oblivious(sym_eq(list(arg.meta["val"].shape), size2)):
        counters["inductor"]["removed_pointless_view_pair"] += 1
        return arg
```

GPU static正例与`-1`正例移除两次view，中间用户负例保留；dynamic/unbacked首维case通过compiled/eager
正确性，FX直接返回输入并记录counter=1。因此8个variant全部有效；只有dynamic case提供数值证据，static
结构case仍不能直接解锁性能。

## 4. T-083：collective分桶

post-grad注册入口位于：

```python
# torch/_inductor/fx_passes/post_grad.py:348-395
if config.bucket_reduce_scatters_fx != "none":
    GraphTransformObserver(...).apply_graph_pass(bucket_reduce_scatter)
if config.bucket_all_reduces_fx != "none":
    GraphTransformObserver(...).apply_graph_pass(bucket_all_reduce)
if config.bucket_all_gathers_fx != "none":
    GraphTransformObserver(...).apply_graph_pass(bucket_all_gather)
```

调用发生在post-grad图改写阶段，早于lowering、scheduler/codegen和autotune。handler位于
`torch/_inductor/fx_passes/bucketing.py`，将多个独立collective的输入打平/拼接，用一次collective传输，
等待完成后再split/view还原；依赖链不得错误合桶。

GPU观察到all-reduce从两个collective改为：

```python
# GPU post-pass，world_size=1
a = aten.view(ar0, [-1])
b = aten.view(ar1, [-1])
packed = aten.cat([a, b])
reduced = _c10d_functional.all_reduce_(packed, "sum", group)
ready = _c10d_functional.wait_tensor(reduced)
left, right = aten.split_with_sizes(ready, [196608, 98304])
return left.view(384, 512), right.view(384, 256)
```

reduce-scatter同样把两个bf16输入拼成294912元素，只保留一次
`reduce_scatter_tensor`；default/custom两种社区方法均通过eager/compiled正确性。all-gather copy-cat证明正例
结构，依赖负例仍保留两次wait。但所有方法都在真实CUDA/NCCL的`world_size=1`运行：它能证明编译合同和
单rank退化语义，不能证明跨rank数据分片、通信量或性能。

GPU结论：6个variant有效，3个单元分母冻结。NPU/性能结论：尚无。下一门禁必须是
`triton_experimental`、真实HCCL、2张可见NPU、2 rank、逐rank输入与输出oracle、collective 2→1计数、
0 graph break/fallback。单rank结果不能解锁通信性能。

## 5. 后续顺序

1. 为7个单元生成NPU最小适配功能入口；后端必须在导入`torch`/`torch_npu`前设为
   `triton_experimental`，每个比较分支fresh process。
2. T-081/T-082先跑原合同；make_fx-only case补同输入NPU数值、shape/stride/alias和目标改图。
3. T-083以真实HCCL 2 rank跑功能；world_size=1仅保留为邻接回归。
4. 功能门禁逐单元生成带源码/输入/handoff哈希的gate。没有合法ON路径、显式产品disable、fallback或
   graph break时不测性能。
5. 通过门禁后才运行已准备的六进程OFF1/ON1/ON2/OFF2/OFF3/ON3性能worker，并生成GPU/NPU对照、
   适配报告或修复报告。

机器复核记录位于：

- `results/current/T-081/gpu_reference_review.json`
- `results/current/T-082/gpu_reference_review.json`
- `results/current/T-083/gpu_reference_review.json`
