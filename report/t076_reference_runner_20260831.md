# T-076 首批 GPU/reference Runner 交付报告

> 报告时间：2026-08-31 20:00 CST（UTC+08:00）
> 状态：`runner-ready-static-validated-await-gpu-execution`
> 动态边界：本机无 CUDA GPU，本报告不包含 GPU/NPU 命中或正确性结果。

## 1. 交付结论

首批 5 个 acceptance units 已形成 manifest-driven GPU/reference 执行入口。执行计划直接复用
T-075 冻结的 13 个 community tests，不复制测试图，不预建 adapter：

| 项目 | 数量/状态 |
| --- | ---: |
| acceptance units | 5 |
| manifest variants | 20 |
| 原生 community cases | 13 |
| 动态 case 覆盖的 variants | 14 |
| static registration-only variants | 3 |
| NPU-only guard variants | 3 |
| adapter/extracted cases | 0 |
| GPU baseline | 未运行 |
| 冻结 denominator | 0 |

14/20 表示“已分配原生动态 case”的 variant 数，不是通过数。另有部分 community case 作为
contract 证据执行，但不被强行绑定到一个并不对应的 variant，例如 `test_exclude_padding` 和
`test_pad_batch`。

## 2. Direct-first 决策

执行顺序通过 schema、plan、runner 三层固定：

```text
原生 community test
→ 保存 return code / skip / stdout / stderr / debug artifacts
→ 判断 reference 是否有效
→ 只在确认设备、backend 或采集接口造成 direct blocker 后讨论 adapter
```

runner 不在失败后自动调用 adapter，也不把 test return code 0 直接等同于有效 baseline。全部 skip、
未发现测试、FX before/after 缺失都会使 `reference_valid=false`。

## 3. 产物

- `upstream/reference_plan.schema.json`：执行计划合同；
- `upstream/reference_plan.yaml`：13 个原生 case、variant 覆盖和 6 个非动态处置项；
- `schemas/reference_result.schema.json`：单 case reference 结果合同；
- `runners/reference_runner.py`：标准库编排器，本身不导入 `torch`；
- `scripts/run_reference_all.sh`：从外部工作目录启动整批的唯一入口；
- `docs/REFERENCE_RUNNER_GPU.md`：GPU 人工更新、执行、重跑、打包和回传说明。

每个 case 单独启动 PyTorch 社区测试进程，配置独立 Inductor cache、`TORCH_COMPILE_DEBUG_DIR` 和
`TORCH_TRACE`。单 case 失败不终止批次；suite 最终通过 `reference_summary.json/.md` 汇总。

## 4. 结果语义

社区测例中的 counter/FileCheck 通常不打印原始数值。runner 因此只在原测试通过时记录
`passed-inside-community-test`，并保留 plan 中的预期断言；`observed_count` 保持 `null`，不从
返回码伪造计数。FX 签名使用 before/after debug 文件内容的归一化 SHA256，不把带时间戳的目录名
纳入签名。

首批 benchmark 标记为 `not-configured`。这是功能 baseline 阶段的有意边界，不是漏测；只有原生
测试与 artifacts 门禁有效后，才允许按同一 contract 增加性能计划。

## 5. 静态验证

所有命令均从 `/home/z50063656/tmp` 发起：

```bash
python -m py_compile \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/reference_runner.py \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_tracker_data.py

python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_tracker_data.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_reference_all.sh \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --validate-only
```

验证结果：5 个 units、13 个 community cases、20 个 variants、14 个动态覆盖、6 个显式非动态
处置全部一致；两条校验路径均声明 `torch_imported=0`，没有运行 GPU/NPU。

## 6. 未完成项

- GPU 机器尚未运行 13 个 direct cases；
- 尚无 direct blocker，因此没有创建 `adapters/`；
- 未冻结 acceptance-unit denominator；
- 未启动 NPU runner、GPU/NPU comparison 或性能阶段；
- T-076 result schema 只覆盖 reference case，后续 P0-E unified GPU/NPU verdict schema 仍需扩展。

下一动作是由 GPU 机器按操作说明执行并回传完整 run 目录。收到 artifacts 后，先判定 direct
有效性和 blocker 类型，再决定是否需要最小 adapter。
