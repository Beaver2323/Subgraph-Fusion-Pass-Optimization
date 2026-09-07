# NPU 修复前后编译证据

> 归档时间：2026-09-08T02:04:47.647073+08:00

## 对照定义

- 修复前：后端修复前：decompose-mm命中后生成错误Triton矩阵乘路径
- 修复后：后端修复后：完整正负例套件回到合法extern矩阵乘路径
- 证据含义：对照forward/backward FX、fusion前后IR与output_code，定位错误自定义Triton GEMM路径并验证修复后的合法lowering。

## 文件怎么读

每个 `run_*/model*` 目录按 Inductor 编译阶段保存五类原始文本：

1. `fx_graph_readable.py`：进入后端处理的可读 FX 图；
2. `fx_graph_transformed.py`：Inductor 图变换后的 FX 图；
3. `ir_pre_fusion.txt`：scheduler fusion 前的 Inductor IR；
4. `ir_post_fusion.txt`：scheduler fusion 后的 Inductor IR；
5. `output_code.py`：最终生成并实际执行的 wrapper/kernel 代码。

不是每个模型都会生成全部五类文件；缺失表示该次编译没有产出该阶段文本，不能补造。
本目录仅归档文本证据，不包含 `.so`、二进制 kernel、cache 或 trace。每个文件的原始绝对路径、
字节数和 SHA256 见 `evidence_manifest.json`。

## 使用边界

“修复前/修复后”只按上面的对照定义解释。产品门禁类修复中，修复前可能是显式重开风险路径，
修复后是默认关闭路径；这类证据证明最终产品行为，而不是两个不同源码 commit 的二分结果。
