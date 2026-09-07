# NPU 修复前后编译证据

> 归档时间：2026-09-08T02:04:47.650166+08:00

## 对照定义

- 修复前：源码修复前：partial-reuse FX改写已发生但NPU生成/执行失败
- 修复后：源码修复后：三类reduction复用图均完成NPU编译与执行
- 证据含义：对照三个模型的FX、fusion IR和output_code，区分pass已命中与后端代码生成真正可执行。

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
