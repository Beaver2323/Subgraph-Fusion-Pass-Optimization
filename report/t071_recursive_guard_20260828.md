# T-071 triton_experimental recursive dict tag guard 验证报告

## 1. 结论

T-071 / TE-GUARD-001 已完成。当前 PyTorch 2.14 上游已经把
`use_recursive_dict_tags_for_guards` 默认设为 `False`，因此 torch_npu 默认
`disable_recursive_dict_tag_guards=True` 在普通启动路径中是幂等 no-op；它只会覆盖用户在
experimental activation 前显式设置的 `True`。默认安全覆盖继续保留，不修改产品源码。

- 静态合同 `12/12`，合成开关/生命周期合同 `6/6`；
- 接受 2 个真实 NPU worker、20 次 eager/compiled 精确比较；
- 深层训练图含 32 个 encoder block 和 32 个 decoder block，首次前后向、3 次 guard replay、
  Python 属性变更后的重编译与第二次前后向均正确；
- gate 开启后递归标签为 False，gate 关闭后显式 True 得以保留，两侧都是 2 个 AOT graph、
  compile peak 同为 `333,824 B`，未复现 `none_dealloc` 崩溃；
- 6 个 CPU worker 全部 `6/6`，三轮每侧 warmup 100、runs 2000。关闭递归标签的 P50 开销
  三轮中位为 `36.57%`，说明该 gate 是稳定性保护而不是性能优化；
- 开关在 experimental activation 时修改 Dynamo 全局 config，`restore_inductor_baseline()`
  不恢复它，activation 后再改 NPU gate 也不会回滚已有值。

最终 verdict 为 `verified-upstream-default-false-safety-override-retained`。不新增 P-026，不提交、
不推送；下一项进入 T-072 / TE-DEC-001。

## 2. 实现与当前上游事实

`apply_npu_overrides()` 在 experimental activation 阶段调用
`_disable_recursive_dict_tag_guards()`。当 NPU gate 为 True 且上游存在目标配置时，它直接执行：

```python
torch._dynamo.config.use_recursive_dict_tags_for_guards = False
```

当前上游 `torch/_dynamo/config.py` 的默认值已经是 False。torch_npu 源码注释仍写着“new in
2.13, default True”，这部分版本描述已经过时；注释记录的历史动机是 speech_transformer 深层
encoder/decoder `ModuleList` 训练在 compiled backward 首次 guard walk 时对 `Py_None` 发生
over-decref，最终 `none_dealloc` abort。当前本地没有 torchbenchmark 包，不能把“最小图未崩溃”
外推为历史模型缺陷已经消失，因此保留与当前上游默认一致的保守策略。

该配置不是 live-read：NPU gate 只在 activation 调用时读取，写入的是 process-global Dynamo
状态。gate=False 表示本次调用不修改当前值，而不是恢复旧值。T-057 已证明 experimental 回切
default 后这项状态仍会泄漏；本轮 inventory 将 binding 更正为
`activation-time-process-global-write`。

## 3. 合成与 CPU guard 合同

合成合同从当前上游默认 False 出发，另外注入 True sentinel，验证了：

1. gate=True 把 True 强制为 False，并对已有 False 幂等；
2. gate=False 保留 True；
3. 缺少 Dynamo import 或配置属性时安全返回；
4. activation 之后把 NPU gate 改为 False 不会恢复之前的 True。

CPU probe 使用 64 个带 `marker=None` 和 Python `offset` 的 block。首次调用只编译一次，修改
`encoder[0].offset` 后精确触发一次重编译，后续稳态调用不再重编译；递归标签开启和关闭的
6 个 worker 均完整通过。

| 指标（三轮中位） | 递归标签开启 | 递归标签关闭 | 关闭路径 paired 开销中位 |
|---|---:|---:|---:|
| mean | `103.546 us` | `142.664 us` | `+37.01%` |
| P50 | `102.950 us` | `141.395 us` | `+36.57%` |
| P99 | `118.680 us` | `162.730 us` | `+38.93%` |

P50 三轮开销分别为 `+30.57%/+36.57%/+65.13%`，三轮方向一致。第三轮存在明显系统抖动，
因此只使用 paired 中位定性为“稳定性保护有可见 CPU guard 成本”，不把最大值当成稳定回退。
该测试测的是 Dynamo compiled-call guard 路径，不是 NPU kernel device time。

## 4. 真实 NPU 训练图

环境为 Ascend910B2、CANN 9.0.1、Python 3.11.15、PyTorch
`2.14.0a0+git8e86e0a`、torch_npu `2.14.0a0+git83cc452`。两个 fresh worker 都从
`/home/z50063656/tmp` 启动，分别使用 device 2/3、P-025 安装 venv、fresh compile cache 和
experimental backend。安装态与当前源码目标函数 SHA256 一致。

每个 worker 对 64-block 深层模型执行训练 forward/backward：首次编译、3 次 guard replay、修改
Python `offset` 后再次编译。输出和输入梯度共 10 组比较全部 bitwise equal，最大绝对误差 0；
每侧 `unique_graphs=2`、AOTAutograd `total=2`、Inductor async cache miss 7，说明 forward 和
backward 都生成并在属性变更后重新编译。两侧未出现 abort、异常或非预期 fallback。

最初 v1 worker 的所有实际比较也完成，但审计脚本错误要求 `unique_graphs>=3`，导致结果文件被
标成失败。AOT 统计对这两个 forward/backward 组合的正确计数是 2；修正断言后重新执行的 v2
结果为两侧 `5/5`，最终聚合只接受 v2，未把审计误判计入产品失败。

首次和属性变更后的编译时长来自不同设备并受编译缓存与负载影响，不用于比较 gate 性能。
compile peak 两侧相同，只说明该最小训练图未观察到内存差异。

## 5. 决策与风险

当前默认路径中，上游本身已经关闭递归标签，所以删除 NPU gate 不会立即产生可测收益；但这会
失去对显式上游 True 和默认值可能变化版本的稳定性兜底。另一方面，强制覆盖用户显式 True 有
CPU guard 性能代价，且是不可恢复的进程全局写入。当前决策是：

- 保留默认 gate，与当前上游 False 一致；
- 不声称最小图证明历史 crash 已修复；
- 不创建只改注释的产品分支，在报告中登记过时注释；
- 把生命周期准确记录为 activation-time process-global write；
- 后续若要允许显式 True，必须先在含 torchbenchmark speech_transformer 的独立进程完成训练
  压测，并同时修复 backend restore 的全局状态隔离，不能只依据 CPU 加速解除保护。

## 6. 证据入口

- 聚合：`results/t071_recursive_guard_20260828/t071_summary.json`（17/17）；
- 静态：`static_contract.json`（12/12）；
- 合成：`synthetic_contract.json`（6/6）；
- NPU：`npu_gate_{on,off}_v2/result.json`（各 5/5、共 20 次比较）；
- CPU：`cpu_r{1,2,3}_{on,off}/result.json`（6 个 worker，各 6/6）；
- scripts：`t071_recursive_guard_{static_contract,synthetic_contract,cpu_probe,npu_probe}.py`、
  `run_t071_recursive_guard_worker.sh`、`aggregate_t071_recursive_guard_results.py`。

## 7. 下一步

Triton experimental 还剩 18 个未关闭 family：P1 14 个、P2 4 个。下一项进入
T-072 / TE-DEC-001 `3D-by-2D matmul fold`，检查布局、动态 shape、反向与 bmm-to-mm 性能。
P-014～P-025 继续只留本地，所有 experimental pass 完成前不统一提交、不推送。
