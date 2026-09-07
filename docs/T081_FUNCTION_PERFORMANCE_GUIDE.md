# T-081 功能与性能测例讲解

> 更新时间：2026-09-08 03:17 CST（UTC+08:00）
> 当前状态：GPU 原生 reference 3/3 cases、10/10 variants 有效；NPU `triton_experimental` 功能 2/2、正式性能处置 2/2 完成。

本批从 5 个旧候选中准备 2 个可运行社区合同，其余逐项保留在文末。冻结 PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。
NPU 全程采用 `triton_experimental`，导入 torch/torch_npu 前选后端。先原生 GPU，阻断后才评审最小适配；显式产品关闭免测。

## GPU 一键执行

```bash
cd /data/z50063656/tmp
bash /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh --task T-081 --gpu 2 --wait-gpu
```

固定回传入口为 `/data/z50063656/tmp/t081-reference-results/latest-text-handoff.json`；超限自动生成分片，上传至 `results/incoming/T-081/`，全部分片与 manifest 同批上传。
一张 GPU 足够。原生 make_fx 结构测试通过只证明注册/改图；编译数值门禁在 NPU 和性能前另补。

## 采集与运行边界

原生 unittest 方法和参数不改写。`native_fx_observer.py` 只包装 joint_graph 入口以保存 before/after；图采集记录 scope、设备及 test_body_modified=false。参数化入口逐项固定，缺例/skip/xfail 不算成功。

直接结构测试调用链是 `社区 test → make_fx → joint_graph_passes → 图节点断言`，不执行设备kernel。编译测试调用链是 `社区 test → torch.compile → AOTAutograd/joint_graph_passes → post_grad_passes → lowering → scheduler/codegen →（存在候选时）autotune`。图改写在lowering/autotune之前；无kernel的视图消除不能伪称内核加速。

本轮run为`reference-20260907T151850+0800-lypgqs3_`，环境指纹为
`ad8df1e2357e5e0351014dc7c3dd578f8ae640b4726d1138bc991cc30da10dc3`。7个原生unittest均执行，
无skip、无adapter。逐case证据边界及FX对照见[GPU复核报告](../report/t081_t083_gpu_reference_review_20260908.md)。

## 动态形状均匀常量折叠与自指形状保护

单元：`AU-joint-graph-constant-fold-uniform-value`。源码入口：`torch/_inductor/fx_passes/joint_graph.py:501`，`constant_fold_uniform_value`。

### 功能测例

- `test/inductor/test_torchinductor_dynamic_shapes.py::TestInductorDynamicCUDA.test_constant_fold_uniform_value_dynamic_cuda`：三个社区原图在static/dynamic及(2,4)/(3,4)输入下相等，恒等图不生成设备kernel。
- `test/inductor/test_torchinductor_dynamic_shapes.py::TestInductorDynamicCUDA.test_constant_fold_uniform_value_self_referential_shape_cuda`：完整原生前后向可编译；自指shape不形成拓扑环，独立测试输入输出一致。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/inductor/test_torchinductor_dynamic_shapes.py::TestInductorDynamicCUDA.test_constant_fold_uniform_value_dynamic_cuda
def full_add_zero(x):
    a = torch.full(x.shape, 1, dtype=x.dtype, device=x.device)
    b = a - 1
    return x + b
```

意图与验证：动态形状均匀常量折叠与自指形状保护。只切换 joint_graph_constant_folding，记录目标入口前后FX；ON必须实质折叠且正确。未形成可区分路径时不计收益。

GPU：dynamic与self-shape两个原生case均通过community eager/compiled正确性；compile-debug post-pass图显示
`full/sub/add`或`full/add/mul`折成恒等返回。旧观察器的`GraphModule.code`缓存曾令入口before/after文本相同，
已做tracker-only最小修复，不能据此误判pass未生效。NPU：`triton_experimental` 下 ON 将
`full/sub/add` 折叠为直接返回输入，数值正确，生成代码由 1 个 Triton kernel 降为 0。

### 性能测例

full_add_zero 原图：(2,4)/(3,4) fp32；性能保留 x + (full(x.shape,1)-1)，dynamic=True。
社区功能图派生的恒等子图；host调用/编译开销是重点，不能解释为全模型吞吐。

OFF/ON：只切换 joint_graph_constant_folding，记录目标入口前后FX；ON必须实质折叠且正确。未形成可区分路径时不计收益。

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## 无损浮点转换链消除与舍入保护

单元：`AU-joint-graph-pointless-convert`。源码入口：`torch/_inductor/fx_passes/joint_graph.py:842`，`pointless_convert`。

### 功能测例

- `test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_convert`：五个参数化方法全部执行；cast节点数2/2/1/2/1，各方法均保留整数往返2节点。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_convert
def fn(x):
    x = torch.ops.prims.convert_element_type.default(x, intermediate_dtype)
    x = torch.ops.prims.convert_element_type.default(x, input_dtype)
    return x
```

意图与验证：无损浮点转换链消除与舍入保护。只禁用 joint_graph.patterns 中 pointless_convert 的 extra_check；ON记录handler前后图差异。不得关闭整个pattern_matcher。

GPU：五个参数化社区方法全部执行，转换节点保留/折叠符合结构断言；该case没有设备数值oracle，只证明
make_fx/joint-pass结构。NPU：已补 `triton_experimental` 数值、dtype/舍入保护、目标改图与无 fallback
证据；ON 将两次 cast 收敛为一次 cast，OFF/ON 最终均为一个融合 kernel。

### 性能测例

原生正例 x[8] fp16 → fp32 → fp16，emulate_precision_casts=True；保留纯cast输出和alias检查。
社区make_fx结构合同派生；先直接joint pass OFF/ON生成图，再以同配置编译测量。编译后可能被下游同等消除，需保留无可区分设备工作结论。

OFF/ON：只禁用 joint_graph.patterns 中 pointless_convert 的 extra_check；ON记录handler前后图差异。不得关闭整个pattern_matcher。

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## Deferred 候选与下一步

- `AU-joint-graph-div-softmax`：与mul合为scaled-softmax合同；原生test_scaled_softmax与nonfinite方法构造CPU张量，common不迁移设备。尚未得到原生GPU阻断/最小设备适配审核记录。
- `AU-joint-graph-mul-softmax`：共享测试同时覆盖mul、反序mul、div、broadcast与非有限值，应合为AU-joint-graph-scaled-softmax，两个旧ID完整保留。CPU执行不能冻结GPU分母。
- `AU-joint-graph-fix-iota-device`：源码788行修正仅被index/index_put使用的CPU iota设备；未找到直接社区断言。constructor mover属于另一pass，不能借其结果代替。

Deferred 项不放入 acceptance_units，不自动重排后续 T 编号，也不进入 GPU suite 分母。其功能/性能阻断都在 manifest 留档，获得新证据后再审核。

## NPU 功能与性能最终结果

两个单元均已通过数值、目标改图、零 graph-break/fallback 门禁，并完成
`OFF1→ON1→ON2→OFF2→OFF3→ON3` 独立进程实测：

- constant-fold：NPU Event p50/p99 改善 29.42%/25.57%，`PERF_IMPROVED`；
- convert：NPU Event p50/p99 变化 -0.04%/+0.82%，`PERF_NEUTRAL`。

完整调用栈、代码框、GPU/NPU 对照、问题修复和证据路径见
[T-081～T-083 NPU 功能、适配与性能报告](../report/t081_t083_npu_function_performance_20260908.md)。
可复现实测命令和 gate 规则见[性能复核门禁](PREPARED_PERFORMANCE_GATE.md)。
