# T-080 GPU reference handoff 复核报告

> 复核时间：2026-09-07 08:32 CST（UTC+08:00）
>
> GPU 运行时间：2026-09-07 00:14～00:18 CST（UTC+08:00）
>
> 结论：13/13 direct cases、13/13 variants 有效，T-080 的 3 个 acceptance units 冻结为 GPU reference；NPU 尚未运行

## 1. 输入与完整性

- handoff：`results/incoming/T-080/manifest.json` 与 `part-0001.json`～`part-0005.json`
- 格式：`1.3 review`，分片文本 handoff
- run ID：`reference-20260907T001455+0800-uv9_464e`
- payload SHA256：`a197df02e0d2b6141ae259fdc69526ddc0888ad4d5179086047ec9b284497f9b`
- 环境指纹：`ad8df1e2357e5e0351014dc7c3dd578f8ae640b4726d1138bc991cc30da10dc3`
- PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`，工作树 clean
- 设备：NVIDIA A100-SXM4-80GB；原生 GPU reference backend 为 `inductor-default`
- 可恢复关键正文：80 份；其余 188 份成功日志、生成代码、IR 或缓存只保留大小和 SHA256

上传时文件名被简写为 `1.json`～`5.json`，与 manifest 中的 `part-0001.json`～`part-0005.json`
不一致；控制节点只做文件名归一化，没有改动分片正文。归一化后导入器验证整包 payload、逐文件
SHA256、case inventory 与路径安全规则，结果为：

```text
handoff_validation=OK run_id=reference-20260907T001455+0800-uv9_464e restorable_text_files=80 code_executed=false
```

## 2. 逐单元 GPU 行为

| Acceptance unit | 社区 cases | GPU 功能结果 | FX/合同证据 | GPU verdict |
| --- | ---: | --- | --- | --- |
| `AU-joint-graph-scatter-upon-const-tensor` | 8 | 8/8 passed | 正例形成 `iota/eq/where` pointwise；三个负例保留 scatter；dtype 与 CE backward 原生断言通过 | 8/8 variants valid |
| `AU-post-grad-prepare-softmax` | 3 | 3/3 passed | max/sub/exp/sum 被替换为 `prepare_softmax_online.default`；signed-zero 与 fast-math 断言通过 | 3/3 variants valid |
| `AU-post-grad-move-constructors-to-gpu` | 2 | 2/2 passed | arange 正例 kernel count=1；index_put 负例保留 CPU constructor 依赖 | 2/2 variants valid |

### 2.1 常量 scatter→pointwise

正例的捕获图包含 selector broadcast 和 pointwise 选择：

```python
# GPU artifacts: cases/REF-scatter-const-3d-native/fx_after.txt
iota = torch.ops.prims.iota.default(...)
eq = torch.ops.aten.eq.Tensor(iota, ...)
where = torch.ops.aten.where.self(eq, val, background)
```

该改写发生在 joint graph 阶段，因此 readable/transformed 捕获点可能都已经位于改写之后。命中结论
同时依赖专属社区测试中的 `num_matches_for_scatter_upon_const_tensor`、访存量和数值断言，不能只用
FX before/after 是否不同来判断。short-index、dense-selector、non-constant-base 三个负例仍可看到
`aten.scatter`/`copy_`，说明 shape、密度和常量 guard 没有被放宽。FP16/BF16 分支保持目标 dtype；
CrossEntropy backward 的梯度和 target metric 也通过。

### 2.2 prepare-softmax→online primitive

三个入口覆盖默认 perf 合同、fast-math 和 signed-zero。正例清楚显示：

```python
# GPU artifacts: cases/REF-prepare-softmax-community-perf-native/fx_before.txt
xmax = torch.ops.aten.amax.default(x, [-1], True)
xsub = torch.ops.aten.sub.Tensor(x, xmax)
xexp = torch.ops.aten.exp.default(xsub)
xsum = torch.ops.aten.sum.dim_IntList(xexp, [-1], True)

# GPU artifacts: cases/REF-prepare-softmax-community-perf-native/fx_after.txt
xmax, xsum = torch.ops.prims.prepare_softmax_online.default(x, -1)
```

fast-math 分支在 online primitive 后继续执行 `log`；signed-zero 分支保留独立 amax 检查，并只把
目标 softmax 前处理改写为 online primitive。社区生成代码和数值断言均通过。

### 2.3 constructor mover

arange 正例的本轮 `fx_before.txt` 与 `fx_after.txt` 都仍显示 CPU `iota`、`device_put` 和 add。这个
debug 捕获点早于后续 constructor mover/codegen，不能把相同 FX 签名误判为 pass 未生效，也不能
把 FX 当作移动完成的直接证据。原生社区用例真实执行并断言 `generated_kernel_count=1`，证明安全
构造器没有形成额外 copy kernel，作为该阶段的计数依据。

index_put 负例在 transformed 图中仍保留 CPU scalar `full` 和 `index_put_` 依赖，且社区原生 codegen
断言确认没有错误生成 GPU scalar constructor。该例的正确性合同是 codegen-only；不能扩写为未执行
的 eager 数值对照。

## 3. 冻结与后续边界

本轮满足同一冻结 PyTorch commit、原生社区入口、无 GPU adapter、无 skip、全部 case
`reference_valid=true`，因此冻结 T-080 的 3 个 GPU denominator。它不代表：

- NPU 已命中、正确或有性能收益；
- 其他 NPU backend 的历史结果可迁移到 `triton_experimental`；
- GPU reference 默认小 shape 已经执行社区完整性能 benchmark；
- constructor mover 的当前 FX 捕获点可以替代后续 kernel/copy 证据。

下一步从 `/home/z50063656/tmp` 启动 fresh process，在导入 `torch`/`torch_npu` 前选择
`triton_experimental`。先执行原生入口；若 prepare-softmax 只被 generic device guard 阻断，单独评审
最小适配。只有目标改写/分解、正确性和 GPU/NPU comparison 都通过后，才能按
`upstream/t080_performance_plan.yaml` 执行同 backend OFF/ON 性能测量。
