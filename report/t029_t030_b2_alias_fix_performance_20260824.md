# T-029/T-030：B2 alias 正确性修复与单 pass 性能

## 结论

- `cat_to_view_pass`：原实现数值正确但把 eager cat 的新 storage 错变成输入 alias。
  全覆盖 identity 分支改为 contiguous clone 后，安装态 NPU 正负例的数值、dtype、
  stride、alias 与图门禁全部通过。三轮 paired p50 改善 2.29%，p99 回退 0.78%，
  每步 task 从 3 降为 1，additional allocated peak 减少 4,195,840 B；结论为
  `supported-neutral-resource-beneficial`，保留修复，不手写额外 Triton。
- `fold_reduce`：clone 修复同样关闭了 alias correctness blocker，但三轮 paired p50
  回退 3.06%，p99 回退 6.72%，allocated peak 不变；两侧都为 1 task，原 size-one sum
  kernel 平均 4.806 us，clone kernel 平均 5.642 us。结论为
  `correctness-fixed-performance-regressed`，下一步应保留原 sum、禁用这条折叠，
  而不是再写一个等价 copy Triton kernel。

## 构建与安装态验证

- 修改前 wheel 已备份到 `artifacts/torch_npu_t029_before_alias_fix.whl`，SHA256 为
  `d0ee10794f8cb63d528c86f27294a2a52a4b8b5f484eb6be53323d22b2157718`。
- T-029 wheel：`torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，
  SHA256 为 `51f484457e555544c171167ff5a652478995f4acc3d749f6accf0a091a00e4df`。
  wheel 内两份修复文件哈希与源码一致；以 `--no-deps --force-reinstall` 安装成功。
- 构建完成 1175 个步骤，只有既有编译告警；临时 `torchgen/packaged` 已恢复为不存在，
  ACL 三个子模块干净。
- 安装态 dynamic-shape FX 测试 33/33 通过。首次受限执行因无法枚举设备报
  `aclInit 507008`；允许访问 NPU 驱动后同一命令通过。该失败是环境中性证据。

## 功能门禁

四项均在物理 NPU 1、default backend、fp16、dynamic compile、fresh cache 下完成：

| case | 图门禁 | 数值 | dtype/stride | eager/compiled alias | 状态 |
|---|---|---:|---|---|---|
| fold_reduce positive | sum 1→0，clone 0→1 | max error 0 | 一致 | false/false | 通过 |
| fold_reduce negative | sum 1→1 | max error 0 | 一致 | false/false | 通过 |
| cat_to_view positive | cat 1→0，clone 0→1 | max error 0 | 一致 | false/false | 通过 |
| cat_to_view negative | cat 1→1 | max error 0 | 一致 | false/false | 通过 |

成功结果位于 `results/t029_alias_fix_20260824/`。按成功证据读取规则，本轮没有读取成功
case 的 `output_code.py`；pre-fix 直返输入证据仍保留在 T-028 结果目录。

## 性能方法与结果

- 形状：fold_reduce `(1024, 1, 1024)`；cat_to_view `(2048, 1024)`。
- baseline 在原 registry 位置观测但跳过目标 pass；candidate 调用原 pass。每侧都验证
  wrapper 恰好调用一次，且计时前后通过数值与 alias 合同。
- 三轮 fresh process，顺序 `B1,C1,C2,B2,B3,C3`；每轮 warmup 10、runs 100。
  每侧第 1 轮额外 profile 10 active steps。采样前后 NPU 1 都无外部进程。

| case | baseline/candidate p50 ms | p50 | p99 | task/step | allocated peak delta | verdict |
|---|---:|---:|---:|---:|---:|---|
| fold_reduce | 0.231855 / 0.238960 | -3.06% | -6.72% | 1→1 | 0 B | performance-regressed |
| cat_to_view | 0.261015 / 0.255040 | +2.29% | -0.78% | 3→1 | -4,195,840 B | neutral/resource-beneficial |

聚合证据：

- `results/t030_b2_alias_performance_20260824/fold_reduce/aggregate/aggregate.json`
- `results/t030_b2_alias_performance_20260824/cat_to_view_pass/aggregate/aggregate.json`

## 失败与中性尝试

- pre-fix 的“直接返回输入”虽快，但违反 eager storage alias 语义，不能作为优化。
- `fold_reduce -> clone` 是正确但性能差的中间方案，不能写成最终优化成功。
- `cat_to_view -> clone` 未通过严格 10% latency 门槛；task 和显存收益单独保留。
- 不继续手写 clone Triton：candidate 已是单个 Triton pointwise kernel，而合法 baseline
  sum 更快；重写同一 copy 不解决已观测瓶颈。

## T-031 最终产品状态

- `fold_reduce` 已收敛为显式 no-op，保留原 sum；`cat_to_view_pass` 继续使用 alias-safe
  contiguous clone。源码覆盖与最终安装态 FX 测试均为 33/33 通过。
- 最终 wheel SHA256：
  `29c3c105453a36d8f2eb648eeb0a2d35cfd0cb871c34697c6aaf17fb1a96a6f5`；wheel 内
  两份相关 Python 文件与源码哈希一致，以 `--no-deps --force-reinstall` 安装完成。
- 最终 NPU 四项均 `npu-compile-complete`、max/mean error 为 0，dtype、stride、alias
  全部匹配 eager：fold_reduce 正负例 sum 均 `1→1`；cat positive 为 cat `1→0`、
  clone `0→1`，negative 为 cat `1→1`。
- 最终结果位于 `results/t031_final_wheel_20260824/`。性能失败的 T-029 clone candidate
  wheel 已单独保存在 `artifacts/torch_npu_t029_alias_safe_clone_candidate.whl`，不会与
  当前安装态 wheel 混淆。
