# T-079 GPU reference handoff 复核报告

> 复核时间：2026-09-07 07:50 CST（UTC+08:00）
>
> GPU 运行时间：2026-09-06 21:21～21:22 CST（UTC+08:00）
>
> 结论：4/4 direct cases、14/14 variants 有效，T-079 的 4 个 acceptance units 冻结为 GPU reference；NPU 尚未运行

## 1. 输入与完整性

- handoff：`results/incoming/T-079/text-handoff.json`
- 格式：`1.3 review`，多行可读 JSON
- run ID：`reference-20260906T212121+0800-898omha0`
- payload SHA256：`dba65b563da64991f556e2b93658b0996dae266fd207e8a93fe4fc6becc024e7`
- 环境指纹：`ad8df1e2357e5e0351014dc7c3dd578f8ae640b4726d1138bc991cc30da10dc3`
- PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`，工作树 clean
- 设备：NVIDIA A100-SXM4-80GB；原生 GPU reference backend 为 `inductor-default`
- 可恢复关键正文：26 份；其余 130 份成功日志、生成代码、IR 或缓存仅保留大小和 SHA256

导入器同时验证整包 payload、逐文件 SHA256、case inventory 和路径安全规则，结果为：

```text
handoff_validation=OK run_id=reference-20260906T212121+0800-898omha0 restorable_text_files=26 code_executed=false
```

## 2. 逐单元 GPU 行为

| Acceptance unit | 社区 case | GPU 功能结果 | FX/合同证据 | GPU verdict |
| --- | --- | --- | --- | --- |
| `AU-joint-graph-bmm-to-mm` | `test_bmm_to_mm` | 1/1 passed | batch=1 为 `squeeze→mm→unsqueeze`；batch=3 保留 `bmm` | 2/2 variants valid |
| `AU-post-grad-cat-slice-cat` | `test_cat_slice_cat_cuda` | 1/1 passed | 三个结构都进入 handler；合法 size 折叠，越界/负 size 由 handler guard fallback | 3/3 variants valid |
| `AU-post-grad-splitwithsizes-cat-replace` | `test_splitwithsizes_cat` | 1/1 passed | 完整同序正例变为 replacement；缺片、异维、重排保留原图 | 4/4 variants valid |
| `AU-post-grad-cat-splitwithsizes-replace` | `test_cat_splitwithsizes` | 1/1 passed | 同边界单用户正例变为 replacement；四个 guard 分支保留原图 | 5/5 variants valid |

### 2.1 bmm→mm

正例捕获到：

```python
# GPU artifacts: cases/REF-bmm-to-mm-native/fx_before.txt
squeeze_dim = torch.ops.aten.squeeze.dim(arg0_1, 0)
squeeze_dim_1 = torch.ops.aten.squeeze.dim(arg1_1, 0)
mm_default = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1)
unsqueeze_default = torch.ops.aten.unsqueeze.default(mm_default, 0)
```

负例 batch=3 捕获到 `torch.ops.aten.bmm.default`。这里 readable 与 transformed 的正例签名相同，
原因是 `bmm_to_mm` 属于 joint-graph 改写，debug 捕获点已经位于该改写之后；不能据此反推 pass 未命中。
社区测试对生成代码中 `mm`/`bmm` 的原生断言已通过，和 shape/FX 一起构成计数证据。

### 2.2 cat→slice→cat

三个 transformed 图都显示：

```python
# GPU artifacts: cases/REF-cat-slice-cat-native/fx_after.txt
cat_slice_cat = torch__inductor_fx_passes_post_grad_cat_slice_cat(
    [arg0_1, arg1_1],
    size=19,
)
```

合法分支首输入宽度为 32；越界分支首输入宽度为 8；负 end 分支的 `size=-1`。这个 FX 节点只证明
pattern 进入 handler，不等同于三个分支都完成折叠。正式解释以 handler 的静态范围 guard 和社区原生
断言为准：合法分支优化，越界及负 end 保持原语义。

### 2.3 split_with_sizes→cat

正例 transformed FX 中原 `split/getitem/cat` 被替换为：

```python
# GPU artifacts: cases/REF-splitwithsizes-cat-native/fx_after.txt
splitwithsizes_cat_replace = (
    torch__inductor_fx_passes_post_grad_splitwithsizes_cat_replace(
        input_=arg0_1,
    )
)
```

缺片、cat 维不同和 getitem 重排三个负例的 transformed FX 与 readable FX 保持原结构，说明 guard
没有被错误放宽。

### 2.4 cat→split_with_sizes

正例 transformed FX 中 cat/split 被替换为：

```python
# GPU artifacts: cases/REF-cat-splitwithsizes-native/fx_after.txt
cat_splitwithsizes_replace = (
    torch__inductor_fx_passes_post_grad_cat_splitwithsizes_replace(
        input_=[arg0_1, arg1_1, arg2_1],
    )
)
```

cat 多用户、split 维不同、split 数量不同、边界不同四个负例均保留 `cat/split_with_sizes`，和社区
计数及 eager/compiled 正确性断言一致。

## 3. 冻结与后续边界

本轮满足同一冻结 PyTorch commit、原生社区入口、无 GPU adapter、无 skip、全部 case
`reference_valid=true`，因此只冻结 T-079 的 GPU denominator。它不提供以下结论：

- 不代表 NPU 已命中或正确；
- 不代表 `default`、DVM 或 MLIR 后端结果可迁移；
- 不代表已经有性能收益；当前 case 的 `benchmark.status` 都是 `not-configured`；
- review 包未嵌入通过 case 的生成代码/IR 正文，需要深度排障时仍应从 GPU 原 run 导出 archive。

下一步在 `/home/z50063656/tmp` 启动独立进程，先选择 NPU `triton_experimental` backend，再依次做
原生入口、必要时的最小设备适配、目标改写证据、正确性和 GPU/NPU comparison。只有功能与命中门禁
通过后，才执行 `upstream/t079_performance_plan.yaml` 规定的 OFF/ON 性能测量。
