# T-078 邻接补证与 T-084～T-086 GPU reference 复核

> 复核时间：2026-09-10T04:20:00+08:00
> 冻结基线：PyTorch `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`、A100-SXM4-80GB、CUDA 12.6、Inductor 默认 CUDA backend。
> 结论：T-084～T-086 共 8/8 cases、11/11 variants 有效，5 个 acceptance units 冻结 GPU 分母；NPU 与性能尚未执行。T-078 新包是有效 codegen 邻接证据，但不是仍缺的 FP16 value=1 数值例。
> 状态说明（2026-09-10T07:33:34+08:00）：上一行是本报告复核时点的历史结论；T-084～T-086 后续 NPU 功能、性能与产品处置均已完成，当前结论见 [NPU 功能、性能、适配与修复报告](t084_t086_npu_function_performance_and_fix_20260910.md)。

## 1. 完整性与 `device_execution=false`

四组输入均通过 handoff payload、逐文件 SHA256、artifact inventory 与 reference summary 一致性校验；
导入过程只解码 JSON/文本，输出 `code_executed=false`。T-085 原上传缺少 `manifest.json`，但 5 个分片
自带相同 source SHA256，序号 1～5、offset 0～196608 连续，逐片哈希和拼接后 215720 字节整包哈希
全部吻合。控制节点据此恢复 manifest，再由标准导入器完整校验。

准备脚本曾打印的 `device_execution=false` 只表示“这次运行的是零设备静态准备校验，没有访问 GPU/NPU”，
不是对后续 GPU handoff 的判定。本批 handoff 的 `execution.command`、CUDA 设备清单和原生社区断言证明
GPU 侧已实际执行；但这仍不代表 NPU 已执行。

| 任务 | run ID | cases | variants | tests | GPU 执行范围 | 冻结结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| T-084 | `reference-20260909T193634+0800-_ghbanb2` | 1/1 | 1/1 | 1 | 真实 CUDA/NCCL，world_size=1 | 冻结单 rank 功能/结构，不证明跨 rank 收益 |
| T-085 | `reference-20260909T194320+0800-1cd_nvn0` | 5/5 | 8/8 | 5 | overlap 为真实 2-rank CUDA/NCCL；其余单卡 | 3 个单元冻结 |
| T-086 | `reference-20260909T195231+0800-n_nc042d` | 2/2 | 2/2 | 2 | 原生 GPUTests 实际 CUDA 编译/数值/mutation | 1 个单元冻结 |

所有 case 均为 `not-needed-direct-valid`，没有 adapter、skip、xfail、unexpected success 或空跑。

## 2. T-078：上传文件名与实际内容不一致

`results/incoming/T-078/BF16-value=1-text-handoff` 的文件名不能作为证据类型。JSON 内唯一实际 case 是：

```text
test/inductor/test_torchinductor.py::GPUTests.test_addcdiv_fma_uses_fma_and_div_rn_cuda
  -> REF-addcdiv-fma-codegen-native
  -> fma-div-rn-codegen-positive
```

该原生 CUDA 测试真实完成编译并通过社区内部断言：`addcdiv_fma_fused=1`；恢复的
`ir_pre_fusion.txt`/`ir_post_fusion.txt` 含 `ops.div_rn`、`ops.fma`，`output_code.py` 含：

```python
# GPU handoff: cases/REF-addcdiv-fma-codegen-native/.../output_code.py
tmp3 = triton.language.div_rn(tmp1, tmp2)
tmp5 = tl.fma(tmp4, tmp3, tmp0)
```

所以它是有效的 CUDA FMA/codegen 邻接补证，但没有执行
`REF-addcdiv-fma-fp16-value1-derived`。T-078 的 `fp16-value1-bitwise-regression` 仍保持
“NPU 已修复真机通过、GPU dtype/value 邻接待回传”，不能被这个误命名文件关闭。

## 3. T-084～T-086 逐单元 GPU 行为

### 3.1 T-084：dedup reduce-scatter

原图含两个相同 `avg` reduce-scatter 及两个 wait；变换后先对输入做 add，只保留一个
reduce-scatter/wait。社区 eager/compiled 数值断言通过。由于测试是 world_size=1，GPU 只证明
post-grad 改写合同和单 rank 退化语义；NPU 必须另用 `triton_experimental` + 真实 HCCL 两 rank
验证数值、2→1 collective 和零 fallback，之后才能测通信性能。

### 3.2 T-085：overlap、partitioned scatter、pointless cumsum

- overlap：真实 2-rank CUDA/NCCL；FX 中 `prims.device_put(..., non_blocking=True)` 被改为
  `False`，原同步 device_put 保持同步，eager/compiled routed output 通过。
- partitioned scatter：三个高争用 `index_put(accumulate=True)` 从直接 51×10 更新改为 64 个分区
  上的 `index_put_` 后再归约；`accumulate=False` 与显式关闭两类负例均不改写。社区虽有性能例，
  其中 1.5 倍阈值注明来自 ROCm MI300X，不能直接作为 CUDA/NPU 收益标准。
- pointless cumsum：一个社区 unittest 内实际编译 11 个整数、布尔、FP16/FP32/FP64及显式输出
  dtype 函数；before 图有 `aten.cumsum`，after 图以 arange/乘常量等表达且数值/dtype 断言通过。

### 3.3 T-086：reinplace index_put

正例覆盖 FP32/FP16：before 为 functional `index_put.default + copy_`，after 为
`index_put_.default` 且删除 copy-back，社区断言 generated kernel 为 1。负例中原 input 仍作为输出，
before/after 均保留 functional `index_put.default`，generated kernel 为 2。两例均比较 CUDA
compiled/eager 和 mutation，证明了正向改写与活跃输入保护。

## 4. 下一阶段

T-084～T-086 现在只完成 GPU 分母冻结。接下来必须按每个单元的 guide，从
`/home/z50063656/tmp` 启动 NPU 功能验证，并在 import `torch`/`torch_npu` 前选择
`triton_experimental`。先运行社区原入口；只有设备/NCCL 阻断时才采用已审核的最小 NPU/HCCL
适配。功能、目标命中/拒绝、实际改图、数值以及零 graph break/fallback 全部通过后，才可运行各自
准备好的六臂 OFF1/ON1/ON2/OFF2/OFF3/ON3 性能流程。

机器可读复核记录：

- `results/current/T-084/gpu_reference_review.json`
- `results/current/T-085/gpu_reference_review.json`
- `results/current/T-086/gpu_reference_review.json`
- `results/current/T-078/gpu_codegen_neighbor_review_20260910.json`
