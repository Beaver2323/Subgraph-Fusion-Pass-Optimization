# 剩余任务推进与一次性 GPU 目标补测

> 更新时间：2026-09-15 20:20 CST（UTC+08:00）。本页是增量记录；准备通过不等于设备通过。

## 1. 当前结论

原 71 个跟踪 ID 全部保留。继 T-112 后，本轮又确认 pattern 17 在冻结版本中是 pattern 15 的
推理注册别名，独立单元从 70 更正为 69；去重时闭环仍47，随后T-106收尾和T-102五项完成，现为54。
剩余15个独立单元：T-098一项、T-103五项、T-104四项（11/12/14/15）、T-105两项（16/20）、T-107三项（28/29/30）。
按缺口分为T-098、10个attention训练合同、15/20两项和16/29两项GPU目标补测；不把同项重复计数。
T-102五项已安装态复验及六臂性能闭环，4改善、1回退；详见[本批交付](../results/current/T-102/README.md)。
22号PERF_REGRESSED，功能修复保留；T-106合计4/4已处置。21号OFF仍被22接替，用户接受该限制，
不签独立计时门禁、不计收益或默认关闭免测。见[21号处置](../results/current/T-106/pattern-21_讲解.md)、[22号原件](../results/current/T-106/pattern-22_讲解.md)。
T-076/T-077 严格历史再认证仍单列，不因去重自动通过。

## 2. 为什么不再要求 GPU 补跑 17

证据来自冻结 PyTorch `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`，不是按名称猜测：

```python
# torch/_inductor/fx_passes/fuse_attention.py:25、1460 附近（节选）
_INFERENCE_ONLY_SFDP_PATTERNS = frozenset([... , "_sfdp_pattern_17", ...])
# 17 不产生训练注册；推理注册将 dropout 固定为 0。
pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
replacement = partialize_and_update_signature(replacement, dropout_p=0.0)
```

```text
# torch/_inductor/pattern_matcher.py:2194、2060、1786 源码重建调用链
gen_register_replacement -> 加载 serialized pattern -> register_replacement(search_fn_pattern=pat)
 -> gm=None -> check_and_add_duplicate_pattern(..., graph=None, skip_duplicates=True)
 -> 同结构的15已经注册 -> 17重复项返回、不新注册
```

15/17 的两份序列化文件 AST 在仅归一导出名称后完全相同，replacement 在 dropout=0 时 AST 也相同，
trace 样例与 extra_check 一致，15 排在 17 前面。已有 GPU 实际观察到15，与该静态链一致。
这只证明冻结版本默认推理注册的归并，不证明非零 dropout 训练受支持，也不消除15的 NPU 数值问题。

离线可复核：`python scripts/review_attention_17_alias.py --check-current`。
完整源码快照与哈希在 [pattern17_alias_review](../results/current/T-105/pattern17_alias_review/)。

## 3. 16/29 补测设计与代码

独立执行器：[attention_target_supplement.py](../runners/attention_target_supplement.py)。
读取冻结原社区函数 AST，源文件哈希不符或修改锚点变化即拒绝；不写 PyTorch 源文件、不改注册或 guard。

| 单元 | 明确改变的内容 | 保持的内容与验收 |
|---|---|---|
| 16 | 推理外增加训练；dropout 0.4 改为 1e-12 | 原FP32、batch4/1、shape、mask、scale；原helper输出/QKV梯度；精确16训练编号 |
| 29 | 全零内部mask改为外部捕获的非零mask | 原FP16/FP32、batch2训练/推理与batch1推理、mask形状和dtype、safe_softmax、双scale；精确29编号及原数值helper |

```python
# runners/attention_target_supplement.py：AST适配的实质变化
# 16：保留非零dropout的图结构；借鉴社区低dropout数值回归方法。
probability.value = ast.Constant(1e-12)
# 29：保留add(mask)，避免原zeros被消去；mask不新增梯度要求。
assignments[0].value = ast.Name(id="supplement_mask", ctx=ast.Load())
```

16 的低概率比较不等于原 p=0.4 的随机分布等价验证；CUDA replacement 可合法保留数学链，不强求 FA。
不覆盖CUDA明确禁用的half/fp32-mask变体。29使用确定性的[-0.25,0.25]非零mask。
两项复用 `_check_common` 的 atol=1e-3/rtol=0.2，数值、梯度、精确编号都须通过；失败如实回传，不重抽种子。
这是 `derived` 补测，`variant_ids=[]`，不能自动回填原生run或据此自动冻结原variant。

## 4. GPU 可复制执行命令

需要包含本轮新增脚本和计划的仓库版本；本页不代表已推送。无需重跑整个 T-105/T-107。

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
cd /data/z50063656/tmp

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-105 --gpu 2 --wait-gpu \
  --case REF-sfdp-pattern-16-dropout-target-derived

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-107 --gpu 2 --wait-gpu \
  --case REF-sfdp-pattern-29-nonzero-mask-target-derived
```

分别按各次输出的 `handoff_upload_input` 回传。仍自动压缩和超限分片；不手抄原始FX，不只传summary。
放在 `results/incoming/T-105/target-supplement/` 和 `results/incoming/T-107/target-supplement/`，
不覆盖旧 `text-handoff.json`。分片时同目录保留 manifest 和全部part。
新包必须单独人工复核；旧原生批次复核器故意不把派生包当原生suite接受。

## 5. NPU 剩余修复

**2026-09-15 11:02后续更新**：条件部署授权已收到；pattern22部署前两臂22场景/23执行均通过，
两个方法已备份并窄改Pass安装包，无候选安装态原例及三个邻接21次Tensor/9次改写通过，安装态边界22场景/23执行通过；性能待验收，正式闭环仍47/69。
pattern2注册候选10次Tensor/4次精确改写通过，尚未部署。
下面的等锁、未授权文字仅是09:05及之前的历史阶段，不代表当前状态；
最新证据见[部署与边界报告](../issues/REF-sfdp-pattern-22-native/部署与边界回归报告.md)。

T-098完整HF32-off原方法的12600秒轮次已超时；default-HF32失败不覆盖。
22号新增 [select-load设备边界执行器](../runners/attention_select_slice_boundary.py)，覆盖
FP32/FP16/BF16、stride2/4/8、首尾lane、共享广播、非零offset、非连续输入与动态shape。
修复前/候选新进程原排在T-098后，本次等待锁超时、未执行设备，需重新调度；未修改产品安装包。
设备数值回归不等于内存sanitizer；未执行完不写PASS。部署仍须授权、备份和安装态原例复验。

其余训练注册与15/20 OFF数值问题继续按原件逐项定位；不套用22修复、不移除邻接pattern制造21收益。
本轮没有因为设备锁等待或GPU待回传就把失败记录改为完成。

## 6. 本轮检查与运行交接

09:05收束：T-098 `return_code=124 / timed-out`，完成84/112个组合，最后进度为169次Tensor比较通过；
其[超时进度原件](../issues/REF-efficient-conv-bn-eval-product-native/precision_runs/native-20260915T002743-tyxjuwzb/progress_at_timeout.json)与同目录run_result.json一并保留。
22号[排队结果](../issues/REF-sfdp-pattern-22-native/boundary_runs/20260915T012916/run_result.json)为
`not-run-lock-timeout / device_execution=false`。下列01:43段落仅记录当时进度，不能作为当前运行状态。

统一零设备检查 `validate_all.py --write-audit` 已通过：297项单元测试，以及计划、归档、矩阵和源码语法检查。
严格历史再认证仍为41项pending、3项exempt，不能把 `tooling_gate=passed` 当成设备或历史通过。

截至01:42:31，T-098 HF32-off完成60/112个组合、120次Tensor断言通过，仍为
`running-not-a-verdict`；原件在 `/home/z50063656/tmp/t098-native-22v6s_le/adapter/`。
22号边界回归仍等待该测试锁，队列记录在
`/home/z50063656/tmp/attention-slice-boundary-nla36v7g/run_result.json`；尚未执行设备，不能填PASS。

本轮使用图模式解单流程，将FX/IR/codegen证据与隔离候选、安装态复验分开。
产品部署授权尚待确认：只在候选及边界回归通过后备份并修改对应的安装态
`triton_experimental` 文件，再用不加载候选的新进程复验；本轮未部署。09:05按用户授权提交推送现有结果，提交号以Git记录为准。
