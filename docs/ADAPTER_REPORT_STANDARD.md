# NPU 最小适配报告规范与索引

> 更新时间：2026-09-07 11:20 CST（UTC+08:00）
>
> 适用后端：`triton_experimental`

适配报告回答“怎样让同一个社区测试合同进入 NPU”，不回答“产品源码怎样修好”。适配器只能处理
测试发现、设备/backend 注入、测试类实例化和证据采集；它不是产品修复，也不能绕过明确产品 gate。

## 1. 与其他报告的边界

| 文档 | 回答的问题 | 必要条件 |
| --- | --- | --- |
| `复现报告.md` | 原生入口发生了什么，如何复现 | 有 NPU 执行或阻断证据 |
| `适配报告.md` | 为什么需要 adapter、偏离了什么、是否保持社区合同 | 实际存在 `npu_adapter.py` |
| `根因分析.md` | 首个产品实现分歧在哪里 | 发现产品差异或回归 |
| `修复验证报告.md` | 产品代码修改后如何证明修复且无邻近回归 | 修改或验证产品候选 |

同一 case 可以同时拥有四份报告。不得把测试入口适配写成产品支持，也不得用 adapter PASS 覆盖
comparison 中的 `EXPECTED_PRODUCT_DIVERGENCE` 或 `NPU_REGRESSION`。

## 2. 强制内容

每份 `issues/<case-id>/适配报告.md` 必须包含：

1. 北京时间时间戳、任务号、acceptance unit、冻结的 PyTorch commit 和实际 backend；
2. community nodeid、原生命令/入口、`NO_TESTS`/skip/device guard 等直接阻断；
3. adapter 代码位置和关键代码框；
4. `adapter_deviation`：逐项解释与原生入口的差异；
5. `preserved_contract`：shape、dtype、stride、dynamic、正负例、counter 和断言如何保持；
6. 从 adapter 入口到社区断言或失败点的最短必要调用链；
7. `product_gate_bypassed=false`，或者明确拒绝把该运行计入正式结果；
8. NPU 结果、正式 comparison verdict、证据路径和适配器不能证明的边界。

代码位置必须写在代码块注释中。调用栈若由源码重建而不是异常 traceback，必须明确标注，不能伪造
动态堆栈。

## 3. 当前覆盖

| 批次 | adapter case 数 | 覆盖状态 | 主要类型 |
| --- | ---: | --- | --- |
| T-076 | 12 | 12/12 | GPU-only suite/harness、NPU backend/device、ATEN fallback 与产品 gate 观察 |
| T-077 | 5 | 5/5 | GPU 测试生成、device guard、精确社区 shape/threshold 镜像 |
| T-078 | 11 | 11/11 | `HAS_GPU` 启动门、原类/原方法实例化、只读 counter/FileCheck 观察 |

逐 case 入口：

- T-076：[mm-plus-mm](../issues/REF-mm-plus-mm-native/适配报告.md)、
  [pad-mm dynamic-M](../issues/REF-pad-mm-dynamic-m-native/适配报告.md)、
  [original ATen](../issues/REF-pad-mm-original-aten-native/适配报告.md)、
  [stride](../issues/REF-pad-mm-stride-native/适配报告.md)、
  [exclusion](../issues/REF-pad-mm-exclusion-native/适配报告.md)、
  [bmm dynamic-batch](../issues/REF-pad-bmm-dynamic-batch-native/适配报告.md)、
  [bmm static-fp16](../issues/REF-pad-bmm-static-fp16-native/适配报告.md)、
  [bmm autocast](../issues/REF-pad-bmm-autocast-regression-native/适配报告.md)、
  [addmm dynamic-M](../issues/REF-pad-addmm-dynamic-m-native/适配报告.md)、
  [addmm bias](../issues/REF-pad-addmm-bias-native/适配报告.md)、
  [addmm contract](../issues/REF-addmm-contract-native/适配报告.md)、
  [symbolic scalar](../issues/REF-addmm-symbolic-scalar-negative-native/适配报告.md)。
- T-077：[Gumbel-max](../issues/REF-gumbel-max-trick-native/适配报告.md)、
  [B2B GEMM](../issues/REF-b2b-gemm-native/适配报告.md)、
  [decompose BMM](../issues/REF-decompose-bmm-native/适配报告.md)、
  [decompose MM](../issues/REF-decompose-mm-native/适配报告.md)、
  [decompose addmm](../issues/REF-decompose-addmm-dynamic-native/适配报告.md)。
- T-078：[addcdiv bitwise](../issues/REF-addcdiv-fma-bitwise-native/适配报告.md)、
  [addcdiv codegen](../issues/REF-addcdiv-fma-codegen-native/适配报告.md)、
  [partial positive](../issues/REF-partial-reuse-positive-native/适配报告.md)、
  [partial negative](../issues/REF-partial-reuse-negative-native/适配报告.md)、
  [addmm core](../issues/REF-unfuse-addmm-core-native/适配报告.md)、
  [expanded](../issues/REF-unfuse-addmm-expanded-native/适配报告.md)、
  [leaf](../issues/REF-unfuse-addmm-leaf-native/适配报告.md)、
  [accumulator negative](../issues/REF-unfuse-addmm-accumulator-negative-native/适配报告.md)、
  [half preserve](../issues/REF-unfuse-addmm-half-preserve-native/适配报告.md)、
  [baddbmm alpha/beta](../issues/REF-unfuse-baddbmm-alpha-beta-native/适配报告.md)、
  [baddbmm core](../issues/REF-unfuse-baddbmm-core-native/适配报告.md)。

direct 成功、明确免测或没有 adapter 的 case 不生成空报告。`tests/test_adapter_reports.py` 强制检查
“每个 adapter 恰有报告”和最小文档合同。

## 4. 新增适配器的提交门禁

```bash
cd /home/z50063656/tmp

python -m unittest discover \
  -s /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/tests \
  -p 'test_adapter_reports.py' \
  -v
```

随后仍需执行 `scripts/validate_all.py`；文档完整不等于 NPU 动态结果有效。
