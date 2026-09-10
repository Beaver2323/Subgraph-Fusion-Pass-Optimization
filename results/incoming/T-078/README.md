# T-078 GPU handoff 接收与现有文件说明

> 更新时间：2026-09-10T04:20:00+08:00

本目录文件名只用于传输导航，验收以 JSON 内 `case_id`、`variant_id`、PyTorch commit、设备环境和
payload SHA256 为准。

| 文件 | 实际内容 | 状态 |
| --- | --- | --- |
| `text-handoff.json` | 原冻结 12 个社区原生 case | 已验收 |
| `BF16-FP16-text-handoff.json` | BF16/FP16 value=2 dtype 邻接 | 已验收 |
| `BF16-value=1-text-handoff` | 实际为 `REF-addcdiv-fma-codegen-native`，不是 BF16/FP16 value=1 数值例 | codegen 邻接有效；文件名误导 |

仍需回传的精确 case 是：

```text
REF-addcdiv-fma-fp16-value1-derived
```

该 case 要求 FP16、`value=1`、compiled/eager bitwise 一致，并确认 `addcdiv_fma_fused=0`；不得用
`REF-addcdiv-fma-codegen-native` 的 FMA/div_rn 正例代替。
