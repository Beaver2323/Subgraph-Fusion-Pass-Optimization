# T-034：B2 第三批 view/copy/where pass 结构与 NPU 编译

## 结论

本批覆盖 `fold_sink_view`、`fold_squeeze`、`fold_to_copy`、`fold_where` 和
`fold_redundant_ops`。新增 10 个 device-independent 测试后，
`test_dynamic_shape_fx_passes.py` 完整结果为 51/51。default-backend NPU 正负例中，除
`fold_to_copy` 在目标 pass 前已被规范化外，其余四条均真实到达、按预期改写或保持，全部
通过数值与完整语义合同。没有产品源码修改，也不需要重建当前 T-031 wheel。

## 环境与方法

- PyTorch `2.14.0a0+git8e86e0a`，torch_npu `2.14.0a0+git83cc452` source-built
  T-031 wheel，CANN 9.0.1，Ascend910B2，物理 NPU 1。
- 每个 case/variant 使用 fresh process、独立 debug/cache、`torch.no_grad()`、dynamic
  compile 与 default backend，均从 `/home/z50063656/tmp` 启动；批次前后 NPU 1 无外部进程。
- registry wrapper 必须只执行一次；同时检查目标节点计数/拓扑、shape、dtype、stride、
  相对每个输入的 storage alias、Python 对象身份和 `requires_grad`。
- fresh Triton host launcher 使用已登记的 audit-only C++20/header shim；它不改变产品 pass，
  也不提升正式无 shim 环境结论。

## 结果

| pass | positive | negative | 完整语义 | 当前结论 |
|---|---|---|---|---|
| `fold_sink_view` | `reshape→relu` 改为 `relu→reshape` | multi-user reshape 仍位于 relu 前 | tensor/tuple 均通过 | 功能通过，性能待测 |
| `fold_squeeze` | matching dim 的 unsqueeze/squeeze 均 `1→0` | different dim 均保持 `1→1` | alias、stride、对象身份均通过 | 功能通过，性能待测 |
| `fold_to_copy` | same-dtype copy 在 pass 前已消失 | dtype conversion 在 pass 前变为 `prims.convert_element_type` | 新 storage 与 dtype 均正确 | reachability-neutral |
| `fold_where` | where `1→0`、clone `0→1` | distinct branches 的 where `1→1` | 新 storage 语义保持 | 功能通过，进入 T-035 |
| `fold_redundant_ops` | reshape/squeeze 均 `1→0` | shape-changing 组合均保持 `1→1` | alias、stride、对象身份均通过 | 功能通过，性能待测 |

全部有效 worker 的 max/mean absolute error 均为 0；这里的 0 只证明数值逐元素相同，最终
判定还依赖上表的完整语义。尤其 `fold_squeeze` 和 `fold_redundant_ops` 的 positive 在图内
直返输入，但编译边界仍重建出 eager 所需的“共享 storage、不同 Tensor 对象”输出，因此
没有发生预先担心的对象身份退化。`fold_where` 的 clone 也正确保持了 eager where 的新 storage。

## 中性失败尝试

首轮审计把编译前目标写成 `aten.view.default`，但动态 pipeline 已将其规范化为
`aten.reshape.default`；同时 sink-view 的首个 positive 使用恒等 view，被更早阶段消除。
这使 4 个 worker 的“图门禁”报错，但输出完整语义均已通过，不是 pass 或 NPU 编译失败。
修正为非恒等 shape 与 reshape 门禁后，4 个 fresh v2 worker 全部通过。首轮目录保留用于
说明失败原因，没有进入最终 verdict。

原始结果位于 `results/t034_b2_view_copy_compile_20260824/`。T-034 不采稳态性能，不能用
首次 compile+run 延迟宣称 pass 有收益。
