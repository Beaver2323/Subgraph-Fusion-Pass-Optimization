# FP16 精度修复证据

> 更新时间：2026-09-09 18:05:53 CST（UTC+08:00）

本目录是 T-078 addcdiv FP16 修复后的追加证据，不覆盖 `../低精度三臂/` 中的修复前事实。

- `summary.json`：旧 6 个 fresh process 汇总，需结合本页纠正说明读取；
- `values/`：`value=0.3/1/2/7.7` 的逐值历史结果，其中 value=1 不计数；
- `guards/`：integer-self 与 tensor-valued value 的不命中结果；
- `value-one/`：已移除补充 pattern 的历史证据；仅用于说明为何不能把 `add(div)` 计为 FMA 命中；
- `value-two/`：社区主形态 `add(mul(div, 2))` 的 FX/IR/codegen。
- `20260909-current/value-one-failure/`：合同纠正后的 value=1 修复前失败，
  mismatch=1176/4096、最大误差=0.015625、counter=0；
- `20260909-fixed/`：value=1 普通 div lowering 修复后的六进程汇总、BF16 邻接和
  value=1 的 FX/IR/output_code；mismatch=0、最大误差=0、counter=0。

当前判定要求：`value=0.3/2/7.7` 必须与 NPU eager 位级一致、`addcdiv_fma_fused=1`；FP16
生成代码必须保留两处显式低精度舍入且不得包含 FMA/div_rn。`value=1` 按社区
bitwise-native 合同必须保持 counter=0；本目录旧 counter=1 工件不再计数。当前
value=1 修复还要求生成代码在 div 后有 1 次 FP16 round、不含 FMA/div_rn，两个 guard
必须保持 counter=0。

解释、源码框与调用栈见 `../../FP16精度修复报告.md`。
