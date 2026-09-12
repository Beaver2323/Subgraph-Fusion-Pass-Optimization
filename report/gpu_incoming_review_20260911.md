# 已上传 GPU reference 复核与证据缺口

> 复核时间：2026-09-11 12:25 CST（UTC+08:00）
> 远端快照：`7a3dbef`（已 fast-forward 拉取 T-102 的全部 11 片）。
> 范围：新批次 T-087～T-112 的已有包，以及 T-077/T-078 补传文件。
> 本报告仅记录 GPU 包的解码、哈希及原文复核；未执行上传代码，未冻结新增分母，未修改历史结论。随后进行的 NPU 实测单列于 [阶段报告](t087_t090_npu_progress_20260911.md)。

## 1. 结论与统计边界

13 个完整新批次共 **40/40 cases、40/40 variants**，原生测试均通过，零 skip、expected failure、unexpected success、超时或空跑；可恢复 **1226 个文本文件**。这证明原生社区测试执行成功，不自动证明计划里每一句人工验收要求都已满足。

这些包涉及 36 个计划单元 ID；其中 T-112 与 T-084 重复，所以不能解释为新增 36 个独立能力。T-091 以及 attention 编号的完整归因仍需补证，也不能直接全部计为正式关闭。

所有完整新包的环境均为 A100-SXM4-80GB、一个可见 CUDA 设备；源码工作树干净，源码实际/预期 commit 与 `torch_git_version` 均为 `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。

## 2. 收件与校验结果

以下路径均相对于仓库；多片包使用同目录 `manifest.json` 导入，单文件包为 `text-handoff.json`。

| 任务 | cases | 计划单元 ID 数 | 收件入口 | 人工复核结论 |
| --- | ---: | ---: | --- | --- |
| T-087 | 3/3 | 2 | `results/incoming/T-087/manifest.json` | 训练重排正负例及运行时设备解析通过；后者不是跨卡执行验证 |
| T-088 | 3/3 | 2 | `results/incoming/T-088/text-handoff.json` | select-cat、split-cat 及 singular 原生断言/FX 可复核 |
| T-089 | 1/1 | 1 | `results/incoming/T-089/text-handoff.json` | view/cat 重排的原生断言与图结构可复核 |
| T-090 | 1/1 | 1 | `results/incoming/T-090/text-handoff.json` | cat 的 positional dim → kwargs dim 有直接 FX 证据 |
| T-091 | 1/1 | 1 | `results/incoming/T-091/text-handoff.json` | 数值通过；缺少 pre-grad 目标 handler 的直接命中/边界图证据 |
| T-096 | 3/3 | 1 | `results/incoming/T-096/text-handoff.json` | E8M0 位操作改写及 1 ULP/历史回归边界通过；NPU 性能 worker 尚未实现 |
| T-102 | 5/5 | 5 | `results/incoming/T-102/manifest.json` | 全部 11 片完整；attention 1～5 原生例通过，逐编号目标边界图待补 |
| T-103 | 5/5 | 5 | `results/incoming/T-103/manifest.json` | attention 6～10 原生例通过；逐编号归因仍需补足 |
| T-104 | 5/5 | 5 | `results/incoming/T-104/manifest.json` | attention 11～15 原生例通过；12 的 training 有源码内精确编号断言 |
| T-105 | 5/5 | 5 | `results/incoming/T-105/manifest.json` | attention 16～20 原生例通过；16 实际是 CUDA inference 专用入口 |
| T-106 | 4/4 | 4 | `results/incoming/T-106/manifest.json` | attention 21～24 原生例通过；22 的特殊 mask stride 不能要求始终生成 SDPA kernel |
| T-107 | 3/3 | 3 | `results/incoming/T-107/manifest.json` | attention 28～30 原生例通过；28 的 dropout 不包含逐值 eager 对齐断言 |
| T-112 | 1/1 | 1 | `results/incoming/T-112/text-handoff.json` | world_size=1 的 2→1 RS 与数值通过；与 T-084 同合同，不能重复计数 |

校验使用仓库 `scripts/import_reference_text.py` 的 `load_input`、`validate_payload`，验证分片/整包、可恢复文件 SHA256、inventory 关系；另外核对实际 case 集合与当前 reference plan、每例 `tests_ran=tests_expected=1`、源码 revision 和环境。只在内存中解码原文，没有执行其中 Python 内容。

收件缺口（截至上述快照）：

- T-098、T-100：目录只有说明文档，没有结果包。
- T-102 已补齐，不再属于收件缺口。run=`reference-20260911T101205+0800-wuph3nq5`，payload SHA256=`b4d5e43c21fb8bd67c3c101bba031460240fa4c98f43acc5a31f80d5f3e32d18`。
- 明确延期的零 GPU-ready 批次不因没有上传包而记为漏测。

## 3. 几个容易误判的功能证据

### 3.1 T-087：重排生效不一定改变算子数量

冻结源码 `test/inductor/test_reorder_for_locality_in_training.py:128` 不仅检查执行成功，还比较 OFF/ON 梯度和 optimizer step 后参数，并验证真正移动了节点：

```python
# test/inductor/test_reorder_for_locality_in_training.py:128（节选）
self.assertEqual(rec_off.calls, 0)
self.assertGreater(rec_on.calls, 0)
self.assertTrue(rec_on.moved)
```

所以不能用 before/after 的算子数量相同否认重排。主开关关闭负例另断言 `rec.calls == 0`。

设备解析例实际入口是 `test/distributed/tensor/test_compile_on_one_rank.py:492` 的 `test_inductor_compiles_under_coor`，真实编译执行并检查 `torch.cuda.current_device()` 及没有固化 CUDA index。它没有 eager 数值比较，也没有在两个设备间复用产物；不能把 runner 的通用 correctness 描述扩大解释为这些验证。

### 3.2 T-090 与 T-091：相邻阶段的 FX 不能代替目标边界图

T-090 的第一组原图与变换图能直接对应：

```python
# T-090 handoff: cases/REF-normalize-cat-default-aten-native/fx_before.txt
cat = torch.ops.aten.cat.default([arg0_1, arg1_1], 1)
# 同 case 的 fx_after.txt（省略类型注解与释放语句）
cat_default_3 = torch.ops.aten.cat.default([arg0_1, arg1_1], dim=1)
```

对应 `torch/_inductor/fx_passes/split_cat.py:1959` 的 `normalize_cat_default_aten`。社区 `test_split_cat_post_grad` 第一组还断言 `normalization_aten_pass=5`、`split_cat_aten_pass=1`；第二组为 3、1。共享 normalization 计数不能外推 clamp/detach/reshape 的独立覆盖。

T-091 源码 `test/inductor/test_split_cat_fx_passes.py:1555` 只比较 `torch.stack([x,y], axis=1)` 的 compiled/eager 输出，没有目标计数断言。回传的 `fx_before.txt` 已是 ATen cat/view，`fx_after.txt` 是 cat/reshape；pre-grad 的 axis→dim 规范化发生在这个采集边界之前。现有证据能证明数值合同，不能单凭这对图证明 `normalize_stack_default` 精确命中。应补目标 handler 计数或该阶段 before/after，不能直接把计划中的“FX 确认 axis 规范化”标为已验收。

### 3.3 T-102～T-107：社区通过不等于每个编号、每个数值分支都验收

冻结源码 `test/inductor/test_fused_attention.py:56` 的 `_check_common` 有三类不同检查：

```python
# test/inductor/test_fused_attention.py:56（条件逻辑节选）
if has_fuse_pattern:
    self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
if expected_fused_attention_pattern is not None:
    self.assertGreaterEqual(
        counters["inductor_pattern_matcher_per_pattern"][expected_fused_attention_pattern], 1
    )
if not has_dropout or override_check_equal:
    self.assertEqual(result1, result2, atol=atol, rtol=rtol)
```

- 通用 `fuse_attention` 大于零只证明某个 attention replacement 生效。多数入口没有传精确编号断言；所读日志/result 也未记录实际 per-pattern counter。编号不能仅凭 case 名或生成了 SDPA 就认定。
- `_test_sdpa_rewriter_12` 明确传 `{True: "_sfdp_pattern_12_training"}`，因此 training 编号归因有原生断言支撑；这个结论不能泛化到所有编号/推理分支。
- 回传的很多 FX before 已包含 `_scaled_dot_product_*_attention`，说明当前采集边界晚于目标融合。after 中 SDPA 仍在不是该编号的原始匹配证据。
- pattern 6 和 28 使用 `has_dropout=True` 且未覆盖默认 `override_check_equal=False`，会运行前后向，但跳过逐值输出/梯度相等断言。其他例应逐入口看参数，不能统一写“所有输出及梯度均与 eager 一致”，也不能因此把社区允许的随机差异直接判为精度缺陷。
- pattern 16 执行 `test_sdpa_rewriter_16_inference_gpu`，CUDA 专门设置 `check_train=False, override_check_equal=True`；本包不证明 training 支持。
- pattern 22 对 mask 最后一维 stride 非 1 的子例设置 `contains=self.device == "cpu"`。CUDA 不强制生成代码保留 SDPA 调用，但仍检查融合计数及数值。部分图重新展开为 bmm/softmax 不等于 pattern 未命中或测试失败。

后续逐编号验收应先补只观测、不改变匹配/门禁的计数或阶段图证据；保留这次原生运行作为基线，不以另造有利图替代它。

## 4. T-112 与 T-084 是同一能力的两层入口

```text
# 冻结源码调用关系
torch/_inductor/fx_passes/post_grad.py:341  config.dedup_reduce_scatters
  -> fsdp.py:167  dedup_fsdp_reduce_scatter
     -> fsdp.py:101  _get_dedup_rs_pass
```

两个 manifest 选择相同 `TestCollectivesInductor.test_dedup_reduce_scatter`、相同开关、相同 2→1 线性 RS 合同。T-084 以 post-grad 开关入口命名，T-112 以内部 pass builder 命名，并非两个独立优化。

本次 T-112 原文确实展示：`wait(RS(a)) + wait(RS(b))` → `wait(RS(a+b))`；源测试精确检查生成代码只有一次 RS 并比较数值。参数 `group_size=1` 也明确在图中，不能由此宣称跨 rank 通信收益。

已在活动 manifest/矩阵将 T-112 指向 T-084 的 canonical 单元，独立贡献为 0；保留本次运行和文件，不新增项目能力分母。重复 NPU/性能入口拒绝执行并指向 T-084。是否复用历史 NPU 结果仍须核对源码 revision、`triton_experimental`、输入、门禁生命周期及测量方法，本轮未重新认证这些历史结果。

## 5. T-077/T-078 补传

- `results/incoming/T-077/REF-decompose-addmm-dynamic-native.json` 校验通过，run 为 `reference-20260911T102337+0800-gfk4dhwz`。它只执行一个动态 addmm 例，属于 `valid-partial-reference-selection`，不是完整套件失败，也不代表 T-076/T-077 全量历史再认证完成。FX 可见 addmm 分解为 unsqueeze/mul/sum/add。
- `results/incoming/T-078/latest-text-handoff.json` 与旧 `BF16-value=1-text-handoff` 的 payload SHA256 完全相同：`4ae45aaf6544d9a6f250029b9dbcd24e597b28158e14a4bc49e97dd9accfe412`。run 为 `reference-20260909T195529+0800-ljhv3d1h`，实际 case 仍是 `REF-addcdiv-fma-codegen-native`。这是重复提交的有效 codegen 证据，不是新的 FP16 value=1 数值证据。

仍缺的 T-078 GPU 例可单独运行，不需要重跑整批：

```bash
cd /data/z50063656/tmp
bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-078 \
  --case REF-addcdiv-fma-fp16-value1-derived \
  --gpu 2 \
  --wait-gpu
```

卡号按分配情况调整；上传本次脚本打印的 handoff 入口，不复用旧文件名猜测内容。

## 6. 下一步顺序

1. 已收束统计与验收口径：T-112 去重归档；T-091、attention 将“原生例通过”和“精确归因待补”分列，不能把 `expected_assertions` 当作全部已实际执行的断言。
2. T-087～T-090 已按原生优先推进 5 个单元的 NPU 社区功能合同，详情见独立阶段报告；设备解析单元、性能门禁及正式 comparison 仍待办。
3. T-096 先验证 NPU 原生阻断及最小适配可行性。当前性能计划明确 `not-implemented`，GPU 通过不意味着性能已准备完成。明确 disable 免测，generic device guard 缺 NPU 不等于明确 disable。
4. T-091、T-102～T-107 的计划已接入 `runners/native_contract_observer.py`：只观察已通过 guard 的精确 handler/replacement 及目标前后图，保留原测试体、设备和断言。观察器本轮仅静态验证，尚未产生新 GPU 归因证据；后续同命令重跑原例即可。真实功能/命中/零 fallback 门禁通过后才运行性能。已有包均为功能 reference，耗时字段不是 ON/OFF 性能收益。
5. 等待 T-098、T-100；T-078 只补缺少的指定例。原上传包保留，不覆盖或删除历史证据。
