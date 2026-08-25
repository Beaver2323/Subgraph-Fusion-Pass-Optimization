# T-036：B2 layout pass 的 alias 缺陷、源码修复与 NPU 闭环

## 结论

本批覆盖 PRE 阶段的 `cat_slice_cat_fold_pass` 与 `pad_slice_fold`。两条 pass 的
正向改写在 Ascend910B2 上都可编译、可执行且数值误差为 0，但原实现都遗漏了
storage alias 语义：

- `cat_slice_cat_fold_pass` 在第一个 `cat` 仍作为图输出时，把第二个独立 `cat`
  替换成第一个对象，导致两个 eager 独立 storage 变成同一 storage、同一对象；
- `pad_slice_fold` 在 slice 直接输出时删掉 `pad`，使 eager 中不别名输入的结果变成
  输入 view，stride 也由 `(6, 1)` 变成 `(4, 1)`。

这说明“最大/平均数值误差都是 0”只证明逐元素值相同，不能证明 pass 语义正确。
T-036 已在产品源码中加入保守保护，构建并以 `--no-deps` 安装新 wheel。60/60 个
device-independent FX 测试及 6/6 个 fresh-process NPU worker 全部通过完整语义合同。
T-036 关闭时两条 pass 的性能尚未测量，因此当时矩阵只登记功能可用；后续 T-037 已按
本报告预留的 gate 完成三轮 paired，并将二者关闭为 `supported-beneficial`。性能数据见
`report/t037_layout_pass_performance_20260825.md`，不能用 T-036 的首次 compile/run 代替。

## 修复内容

修改文件：

- `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py`
- `test/_inductor/test_dynamic_shape_fx_passes.py`

具体规则：

1. `cat_slice_cat_fold_pass` 只有在第一个 `cat` 的全部用户恰好是待删除的 slice
   节点时才折叠。这样第二个 `cat` 被第一个替代后，不会让可观察的两个独立结果合并。
2. `pad_slice_fold` 只有在每个 slice 的所有直接消费者都能被保守证明会物化新 storage
   时才删掉 pad。当前仅允许显式的非原地 elementwise/activation/matmul 消费；图输出、
   view、未知算子和原地操作全部保持原图。
3. 没有用额外 `clone` 掩盖错误，也没有引入 Triton。这里的正确修复是缩窄不安全的
   rewrite capability，而不是在改写后补一次无条件拷贝。

源码中 `_all_users_materialize_new_storage` 位于当前文件约 88 行，cat 的用户集合保护
位于约 156 行，pad 的消费者保护位于约 255 行。行号会随源码演进变化，应以函数名和
测试名检索为准。

## 测试合同

新增/扩展 9 个直接测试：

| pass | 正例 | 负例/语义边界 |
|---|---|---|
| `cat_slice_cat_fold_pass` | 完整连续覆盖时 `cat 2→1`、slice `2→0` | 覆盖有 gap 时保持；第一个 cat 可观察时保持；eager 两输出 storage 独立 |
| `pad_slice_fold` | slice 后接 `relu` 时 pad `1→0` | slice 触及 padding 时保持；slice 直出或接 view 时保持；eager slice 不别名输入 |

完整测试命令从 `/home/z50063656/tmp` 启动，结果为：

```text
Ran 60 tests in 1.554s
OK
```

修改的产品文件和测试文件通过 `lintrunner`，结果为 `ok No lint issues.`。

## 修复前 NPU 证据

环境为 PyTorch `2.14.0a0+git8e86e0a`、torch_npu
`2.14.0a0+git83cc452`、CANN 9.0.1、Ascend910B2、物理 NPU 1。所有 worker
都从 `/home/z50063656/tmp` 启动，使用 dynamic/fullgraph/default backend、独立缓存和
registry wrapper。

| case | pass 前后 | 数值 | 失败的非数值语义 |
|---|---|---|---|
| cat positive | cat `2→1`，slice `2→0` | max error `0` | 无，安全正例 |
| cat gap negative | cat `2→2`，slice `2→2` | max error `0` | 无，正确保持 |
| cat observable alias | cat `2→1`，slice `2→0` | 两输出 max error 均 `0` | 跨输出 storage alias `false→true`，对象身份 `false→true` |
| pad positive + relu | pad `1→0` | max error `0` | 无，relu 物化新 storage |
| pad touches padding | pad `1→1` | max error `0` | 无，正确保持 |
| pad direct output | pad `1→0` | max error `0` | stride `(6,1)→(4,1)`，相对输入 alias `false→true` |

因此两条修复前实现不能判定为“可用”。原始结果保留在
`results/t036_b2_layout_alias_compile_20260825/`。

## 修复后 NPU 证据

候选 wheel：

```text
torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl
SHA256 d745cf3afd6a2859a68d6c31dd02a46498264e82dedff34d726c2be2609c6b9d
```

它由当前 torch_npu 源码构建，并通过 `pip install --no-deps --force-reinstall` 安装。
运行时 `torch_npu.__file__` 指向 Conda 环境的 site-packages，且从安装文件确认三处保护
均存在。T-031 旧 wheel 以 SHA256
`29c3c105453a36d8f2eb648eeb0a2d35cfd0cb871c34697c6aaf17fb1a96a6f5`
保存在 `artifacts/torch_npu_t031_before_t036_layout_alias_fix.whl`，可回滚。

| case | 修复后 pass 前后 | 完整语义 |
|---|---|---|
| cat positive | cat `2→1`，slice `2→0` | 通过，输出 stride `(5,1)`，不别名任一输入 |
| cat gap negative | cat `2→2`，slice `2→2` | 通过 |
| cat observable alias | cat `2→2`，slice `2→2` | 通过；两输出仍为独立 storage/对象 |
| pad positive + relu | pad `1→0`，slice 保留 | 通过，输出不别名输入 |
| pad touches padding | pad `1→1`，slice `1→1` | 通过 |
| pad direct output | pad `1→1`，slice `1→1` | 通过；stride 保持 `(6,1)`，不别名输入 |

六项 max/mean absolute error 都是 0，且 shape、dtype、stride、`requires_grad`、
相对输入 alias、对象身份和跨输出 alias 矩阵全部通过。最终原始证据位于
`results/t036_b2_layout_alias_fix_header_20260825/`。

## 失败和中性尝试

1. 修复前两个 alias worker 都以 `npu-compile-semantic-failed` 结束。这是实际产品语义
   缺陷，不是测试误报；它们被保留为修复动机，不能并入最终通过数。
2. 修复后首次批处理使用 `set -e`，被 `env.sh` 内部允许失败的探测命令提前终止，尚未
   启动 worker；去掉 shell 的全局 `set -e` 后恢复。这不是 pass 或 NPU 失败。
3. 首个修复后 cat-alias worker 把 `CPATH` 错指到 editable 源码的 `torch/include`，fresh
   launcher 因找不到 `ATen/ATen.h` 失败。pass observer 已显示图保持 `2→2`，但没有设备
   执行，故不算通过。改为已安装 wheel 的 `site-packages/torch/include` 后，在全新目录
   完整通过。该失败保留在 `results/t036_b2_layout_alias_fix_20260825/`。

## 当前结论与下一步

- 功能：两条 pass 在保守 capability gate 下可用，安全正例继续优化，风险场景保持原图。
- 精度：代表 fp16 contiguous dynamic/fullgraph case 数值误差为 0；更重要的是完整 alias
  与 stride 合同也通过。
- 性能：T-036 本身未执行 paired；后续 T-037 已确认 cat-slice-cat/pad-slice p50 分别
  改善 24.00%/31.35%，两条均为 `supported-beneficial`。
- Triton：当前问题是 FX rewrite 语义边界，不是 NPU 算子缺失；手写 Triton不会替代
  必须先正确处理的 alias 合同。

下一任务 T-037 已按上述安全正例完成三轮、交替顺序、fresh-process 的单 pass paired：
baseline 仅关闭目标 pass，candidate 启用目标 pass；warmup 10、runs 100，并记录了
p50/p99、task、allocated/reserved peak。最终数据与判定见 T-037 报告。
