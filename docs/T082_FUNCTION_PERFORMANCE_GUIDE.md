# T-082 功能与性能测例讲解

> 更新时间：2026-09-07T22:36:49+08:00
> 准备状态：源码合同已审核，GPU/NPU 均未执行；静态校验不是运行结果。

本批从 4 个旧候选中准备 2 个可运行社区合同，其余逐项保留在文末。冻结 PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。
NPU 全程采用 `triton_experimental`，导入 torch/torch_npu 前选后端。先原生 GPU，阻断后才评审最小适配；显式产品关闭免测。

## GPU 一键执行

```bash
cd /data/z50063656/tmp
bash /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh --task T-082 --gpu 2 --wait-gpu
```

固定回传入口为 `/data/z50063656/tmp/t082-reference-results/latest-text-handoff.json`；超限自动生成分片，上传至 `results/incoming/T-082/`，全部分片与 manifest 同批上传。
一张 GPU 足够。原生 make_fx 结构测试通过只证明注册/改图；编译数值门禁在 NPU 和性能前另补。

## 采集与运行边界

原生 unittest 方法和参数不改写。`native_fx_observer.py` 只包装 joint_graph 入口以保存 before/after；图采集记录 scope、设备及 test_body_modified=false。参数化入口逐项固定，缺例/skip/xfail 不算成功。

直接结构测试调用链是 `社区 test → make_fx → joint_graph_passes → 图节点断言`，不执行设备kernel。编译测试调用链是 `社区 test → torch.compile → AOTAutograd/joint_graph_passes → post_grad_passes → lowering → scheduler/codegen →（存在候选时）autotune`。图改写在lowering/autotune之前；无kernel的视图消除不能伪称内核加速。

## 互逆permute消除与中间用户保护

单元：`AU-joint-graph-pointless-permute-pair`。源码入口：`torch/_inductor/fx_passes/joint_graph.py:957`，`pointless_permute_pair`。

### 功能测例

- `test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_permute_pair`：原生GPU make_fx图2→0节点；中间permute另有输出时保留2节点。
- `test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_permute_pair_3d`：三维同样消除互逆置换、保留多用户。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_permute_pair
def f(x):
    x = aten.permute.default(x, [1, 0])
    x = aten.permute.default(x, [1, 0])
    return x
```

意图与验证：互逆permute消除与中间用户保护。仅屏蔽pointless_permute_pair注册；先直接joint pass保存OFF2节点/ON0节点，再编译；检查输出stride、alias、数值

GPU：等待原生运行，不预判命中。NPU：等待 `triton_experimental` 同合同验证；原始断言、图、设备实现差异将写回 result 与适配/修复报告。

### 性能测例

原生(15,7) fp32，[1,0]两次permute；不增加pointwise或人为拷贝。
视图元数据优化子图；可能没有设备kernel，主要测编译和host调用，不宣称减少HBM通信。

OFF/ON：仅屏蔽pointless_permute_pair注册；先直接joint pass保存OFF2节点/ON0节点，再编译；检查输出stride、alias、数值

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## 互逆view消除、-1与unbacked动态形状

单元：`AU-joint-graph-pointless-view-pair`。源码入口：`torch/_inductor/fx_passes/joint_graph.py:936`，`pointless_view_pair`。

### 功能测例

- `test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_view_pair`：2→0节点，保留中间用户2节点，[-1,7]也消除。
- `test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_view_pair_dynamic_shapes`：mark_unbacked首维，compiled输出相等，removed_pointless_view_pair=1。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_pointless_view_pair
def f(x):
    x = aten.view.default(x, [3, 5, 7])
    x = aten.view.default(x, [15, 7])
    return x
```

意图与验证：互逆view消除、-1与unbacked动态形状。只屏蔽pointless_view_pair注册；pointless_view单节点handler不变；OFF/ON保存精确前后图、dtype/shape/stride/storage关系

GPU：等待原生运行，不预判命中。NPU：等待 `triton_experimental` 同合同验证；原始断言、图、设备实现差异将写回 result 与适配/修复报告。

### 性能测例

原生x[15,7] fp32 →view[3,5,7]→view[15,7]；性能主例static，dynamic回归由功能suite门禁。
社区互逆view元数据子图；不人为补计算来制造设备性能，动态路径作为功能回归。

OFF/ON：只屏蔽pointless_view_pair注册；pointless_view单节点handler不变；OFF/ON保存精确前后图、dtype/shape/stride/storage关系

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## Deferred 候选与下一步

- `AU-joint-graph-pointless-view`：test_pointless_view_pair直接验证的是pair handler，不能再计单view。保留独立单view注册候选，待直接社区证据。
- `AU-joint-graph-remove-noop-ops`：joint_graph调用的是post_grad.remove_noop_ops。complex conj是保护负例，uneven-sharding是2rank SPMD保护旁证；尚缺可独立归属的正例/性能合同。pre_grad remove_noop测试不是本pass。

Deferred 项不放入 acceptance_units，不自动重排后续 T 编号，也不进入 GPU suite 分母。其功能/性能阻断都在 manifest 留档，获得新证据后再审核。

## 性能执行准备边界

性能命令、原件绑定字段和2rank要求见[性能复核门禁](PREPARED_PERFORMANCE_GATE.md)。

`runners/t081_t083_performance_worker.py` 提供目标级工作负载、OFF/ON控制、数值/图门禁和原始计时采集。`scripts/run_prepared_performance.py` 检查每单元经复核的功能门禁记录后启动6个独立进程；当前没有该记录，因此拒绝正式性能测量。
门禁记录必须绑定 acceptance_unit_id、backend、PyTorch commit、功能报告sha256、GPU与NPU通过状态、显式disable处置、fallback/graph-break以及待测输入。未提供/不一致时停在准备状态。
