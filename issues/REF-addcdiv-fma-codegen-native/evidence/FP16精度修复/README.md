# FP16 精度修复证据

> 更新时间：2026-09-08 09:16:00 CST（UTC+08:00）

本目录是 T-078 addcdiv FP16 修复后的追加证据，不覆盖 `../低精度三臂/` 中的修复前事实。

- `summary.json`：6 个 fresh process 的正式汇总；
- `values/`：`value=0.3/1/2/7.7` 的逐值结构化结果；
- `guards/`：integer-self 与 tensor-valued value 的不命中结果；
- `value-one/`：分解器消去乘法后，补充 `add(div)` pattern 的 FX/IR/codegen；
- `value-two/`：社区主形态 `add(mul(div, 2))` 的 FX/IR/codegen。

判定要求：四个标量值必须与 NPU eager 位级一致、`addcdiv_fma_fused=1`；FP16 生成代码必须
保留显式低精度舍入且不得包含 FMA/div_rn；两个 guard 必须 `counter=0`。

解释、源码框与调用栈见 `../../FP16精度修复报告.md`。
