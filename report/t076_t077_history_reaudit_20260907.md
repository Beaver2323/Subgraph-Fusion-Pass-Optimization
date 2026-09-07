# T-076 / T-077 严格历史核验与最短补证路径

> 更新时间：2026-09-07 22:35 CST（UTC+08:00）
> 规则：`2026-09-07.1`；校验器：`2.1.0`。
> 本轮：只读原始证据、复核冻结源码、修复审计工具；没有新跑 GPU/NPU，没有修改产品源码或历史 verdict。

本轮已完成现存证据的核验和缺口定位。**历史完整再认证仍为 pending**，这与已有功能/性能处置完成属于两个口径。
固定机器可读入口为 [`results/audits/latest.json`](../results/audits/latest.json)，最终执行时会追加独立时间戳文件。
旧报告的“GPU 原件都未收到”和“41 项都待人工查”已不适用于当前状态。

## 1. 哪些已核对，哪些仍无法认证

| 项目 | 本轮实际证据 | 剩余缺口 |
| --- | --- | --- |
| GPU 关键正文 | T-076 80 份、T-077 68 份；24/24 case 的 result、inventory、FX before/after 已逐份验哈希 | 24 份非空 stderr 被 review profile 省略；24 份空 stdout 由已绑定 bytes=0/空 SHA256 确定为空，无须重传 |
| GPU 冻结源码语义 | 24 个方法及相关 helper 均读取精确 commit 的 Git blob；19 项在声明范围内具备相应断言 | 2 个辅助 case 无数值比较；3 个 decompose case 的输入梯度未由社区 oracle 对齐，详见下节 |
| NPU 原始运行 | 明确选取 19 份可读最终/候选 adapter JSON、相应日志和生成代码；另读取 1 份修复前 MM 梯度诊断 | 10 个 unit 都缺完整的“运行时实际加载源码/补丁 + 三库 revision + 导入前后端”绑定；当前文件哈希只能证明本次读到的内容 |
| T-077 六轮性能 | 4 个 raw summary 登记哈希一致；24 份 worker JSON 可读；逐项 OFF/ON/counter/正确性/输入合同检查；4/4 独立复算三轮中位数与改善百分比一致 | 无逐次时延样本；完整 torch_npu/Triton revision、进程启动/PID/运行脚本绑定不全；Gumbel 还缺逐 worker backend 字段 |
| T-077 B2B 能力网格 | 12 份 capability JSON、日志与生成代码可读；独立重数 8 命中、4 拒绝、0 模板获选 | 精确运行态溯源不全；12 点不能扩写为90点全网格或完整模型性能验收 |
| T-076 历史性能 | 找回 mm_plus_mm 5 份原始 sweep JSON，共54条 worker记录；T-058 addmm 6份原始性能JSON | 无完整新规则计时与运行态溯源；代表 workload 不等于社区功能case；addmm历史审计启用/P-018候选不能等同当前默认安装态 |
| pad 性能 | 3 个显式关闭免测处置保持有效 | 不补 ON，不重新绕过产品关闭 |

`pending=41、exempt=3` 是24个GPU case、10个NPU unit、7个性能处置的**严格总门禁状态**，不是41个缺失文件。
现在可从 `component_counts` 看到24项关键正文通过、19项源码语义通过、4项性能复算通过等已完成部分。
本轮读取并记录552个证据位置（运行文件、仓库文件、handoff正文、冻结Git blob）；不是552次设备测试。

## 2. 冻结源码核验发现的实际 oracle 边界

所有源码来自 PyTorch `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。审计记录包含完整文件SHA256、方法SHA256和行号；不依赖当前dirty工作树内容。

### 2.1 原始 ATen 名称保留：证明代码生成，不证明输出数值

```python
# 文件：test/inductor/test_pad_mm.py:569
# 方法：PadMMTest.test_original_aten_preserved_pad_mm（冻结源码摘录）
opt_fn = torch.compile(fn, mode="max-autotune")
ret, code = run_and_get_code(opt_fn, *args)
self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
FileCheck().check("def triton_tem_fused_mm").run(code[0])
```

这里没有把 `ret` 与 `fn(*args)` 比较。GPU test通过可以证明原始算子名称进入模板名字，不能单独证明数值对齐。
该case是 pad-mm辅助回归，`variant_ids=[]`；不能把它提升为新的独立动态分母。
补充数值需要新增注明来源的同输入比较，不能改写旧 `reference_result.json` 声称原生测例原本就做了该断言。

### 2.2 exclude-padding：证明cache-key区分，不证明矩阵结果

```python
# 文件：test/inductor/test_pad_mm.py:452
# 方法：PadMMTest.test_exclude_padding（冻结源码结构摘录）
mm(torch.rand(size, device=GPU_TYPE), torch.rand(size, device=GPU_TYPE))
local_cache = get_pad_cache().get_local_cache()
self.assertEqual(len(local_cache), 2)
FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(repr(local_cache))
```

后续 `(a+1) @ b` 分支继续检查缓存键，仍未比较输出。其意图是缓存策略；NPU adapter额外做过输出正确性，不能逆向证明GPU原生test已做该检查。
相比之下，`test_pad_batch` 在方法末尾显式调用 `torch.allclose(res2, bmm_expected_result)`，本轮已确认它有数值oracle。

### 2.3 decompose的forward有效，input gradient比较为空缺

```python
# 文件：test/inductor/test_decompose_mem_bound_mm.py:113
# 方法：TestDecomposeMemMM.compare_gradients（冻结源码摘录）
ref_grad = {key: param.grad for key, param in module.named_parameters()}
res_grad = {key: param.grad for key, param in traced.named_parameters()}
self.assertTrue(self.compare_dict_tensors(ref_grad, res_grad, rtol=self.rtol, atol=self.atol))

# 文件：test/inductor/test_decompose_mem_bound_mm.py:34、42
# MyModule2/MyModule3 的 forward 只接收 mat1/mat2 并调用 bmm/mm，没有 Parameter。
```

必要调用链为：

```text
test_decompose_bmm / test_decompose_mm / test_decompose_mm_mixed_precision
  → compare_pred → assertEqual(eager, compiled)       # 有效forward oracle
  → ref.sum().backward() / res.sum().backward()      # backward实际执行
  → compare_gradients → named_parameters()          # 两侧均为空
  → compare_dict_tensors({}, {}) → True             # 不能证明mat1.grad/mat2.grad对齐
  → decompose counter检查                          # 证明预期改写次数
```

三个case能证明forward正确、backward可执行及counter符合预期，不能证明输入梯度数值一致。
NPU最终adapter的left/right梯度数值检查是额外而有效的证据，但它不补足旧GPU运行缺少的oracle。
此缺口不是“GPU梯度错误”的证明；要增强合同须用相同输入、独立leaf拷贝和同一upstream函数做新的对齐测例并独立记账。

## 3. 后端与历史性能的核验结论

T-076 mm_plus_mm 原始54条记录均声明 `backend=triton_experimental`，实际pass观测的
`active_config_snapshots` 也显示该后端。compile上下文退出后记录的 `effective_config.npu_backend=default`
是不同生命周期的采样，不能据此把整个历史判为default运行；本轮保留两种原始字段及其差别。
但这些记录主要提供host计时，没有完整的同规则Event逐样本证据和实际加载源码绑定。

T-058 addmm的6份原始JSON没有独立backend字段；旧报告明确记载审计用恢复上游check的启用方式，
部分wrapper marker也不能单独证明导入生命周期。该历史仅保留为候选证据；遵守当前明确关闭免测要求，
本轮没有重跑这条被关闭的ON路径。

T-077 decompose worker已有 `environment.torch_git` 完整commit。工具现在按原值映射为
`torch_commit`，不再误报该字段缺失。`generated_at` 是结果生成时间，不能伪装成进程启动时间。

Gumbel的45.79%/45.75% host和46.69%/46.50% Event改善，已经由原始六轮统计值独立复算一致。
这完成“汇总公式是否算对”的核验，但缺失的100次原始样本不能从p50/p99反算，完整运行态也不能从当前脚本补造。
因此保留历史收益数字，同时保留严格性能再认证pending。

## 4. GPU现在最短的补证操作：只复制非空日志

先pull包含本次脚本的tracker版本，再在GPU执行以下命令。脚本只读取明确的旧run，不使用latest，不启动GPU计算。
请在旧run仍保留时补证，不要重新运行后覆盖旧目录。

```bash
# 文件：scripts/export_history_logs.py
cd /data/z50063656/tmp
python "${TRACKER_ROOT}/scripts/export_history_logs.py" \
  --task T-076 \
  --output /data/z50063656/tmp/t076-history-logs.json

python "${TRACKER_ROOT}/scripts/export_history_logs.py" \
  --task T-077 \
  --output /data/z50063656/tmp/t077-history-logs.json
```

默认使用 `reference-20260901T180826+0800` 与 `reference-20260902T125636+0800`；如果旧run被移动，
用 `--run-dir /实际旧run路径` 指定。脚本校验每份日志的原inventory哈希，拒绝修改过的日志，输出采用多行JSON。
超过64KiB时自动拆为32KiB payload的分片，并打印实际应上传的入口；输出文件和分片目录已存在时拒绝覆盖。

上传位置：

| 脚本输出 | 仓库位置 |
| --- | --- |
| 未分片的JSON | `results/incoming/T-076/history-logs.json` 或 `T-077/history-logs.json` |
| 已自动分片 | `results/incoming/T-076/history-logs/manifest.json` 与全部part文件；T-077同理 |

同一任务只保留一种补证入口。现有关键FX handoff保留，**无需再复制148份正文**。
控制节点的 `audit_history.py` 会自动合并、检查原run身份和每份日志原SHA256，再运行严格unittest parser。
此工具生成的是日志补证格式，使用历史审计器读取，不使用普通reference handoff导入器单独恢复。

补日志可解决24项“原始日志重解析”子检查；它不能填上5项源码oracle边界，也不能恢复NPU安装态和性能原始样本。
旧审计工具原本还要求全inventory可读；本轮没有降低该要求。当前仍有T-076 214项、T-077 273项非空archive正文缺失，
其中24项是上述stderr，另外463项是debug/code/IR等登记工件。只补日志后，严格总状态仍为pending。
完整archive对保存历史全量证据有价值，但不应把它与“已读取FX”混为一谈，也不应要求用户为只重解析日志反复传全部cache。

## 5. 本机继续做什么，哪些无法补成历史事实

1. 已落实：直接读取split/review，24份FX链路验哈希；冻结源码语义审阅；本机NPU原件盘点；旧性能文件定位；四项汇总独立复算；自动接入日志补证。
2. 可补历史原件：GPU stderr及剩余archive；若存在当时运行命令/安装文件快照/worker脚本副本，可逐项绑定到原run。当前新生成的快照不能代替这些原件。
3. 无法逆向恢复：未落盘的逐样本时延、未记录的进程开始/PID、无法关联的实际安装文件。应新增同合同、导入前固定后端的独立运行并完整采集；新结果单独认证，旧历史状态不伪造为passed。
4. GPU的2个数值辅助oracle和3个input-gradient oracle需要新补充测例；保持原生测试作为原始reference，新增断言注明最小增强范围。
5. 显式关闭的pad和默认安装态addmm不为追求pending清零而补ON性能。已验证候选与产品安装态分开记录；T-077小MM修复候选仍不自动变为已合入产品。

本轮没有因这些不可追溯字段启动盲目NPU重跑。重跑能创建新证据，不能证明旧进程当时加载了什么。

## 6. 校验与固定入口

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_all.py --write-audit
```

本轮历史专属零设备回归27项通过，覆盖真实handoff身份/哈希拒绝、日志补证编码和路径边界、空stdout免传、
缺archive仍独立重解析日志、torch_git字段映射、损坏逐样本与聚合篡改检测；历史 `results/current/` 文件哈希保持不变。
严格入口 `--require-history-complete` 仍应退出3，而不是把完成本轮核验写成“所有旧证据均已完整再认证”。
