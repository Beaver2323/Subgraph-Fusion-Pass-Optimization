# Attention 性能测例的执行阶段与 dropout 合同复核

> 更新时间：2026-09-15 19:49 CST（UTC+08:00）。本报告解释准备与执行约束；各编号实测结果另列，不能整体视为通过。

本轮T-102修订：1号推理OFF被3号dropout=0推理图接替；1/2/5改测本次修复对应的training前向合同。
2号性能入口另发现scale参数名遗漏，按原Python标量guard补齐具名占位转换。均不放宽OFF/ON门禁，
旧失败原件保留，见[合同修订](../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)与[输入适配](../issues/REF-sfdp-pattern-2-native/性能输入适配报告.md)。
21号现已获用户单独接受归因受限结项（不计实测/收益/默认关闭免测）；17已去重归并15。下段保留早期执行背景。

当前pattern13已完成独立OFF/ON与六臂性能，`PERF_REGRESSED`，
详见[结果及学习说明](../results/current/T-104/pattern-13_讲解.md)。其余编号继续按门禁逐项推进。
pattern15先发现trace用零维scale占位被实际Python标量guard合法拒绝；具名inv_scale在compile外转换后，
新进程OFF出现400/1024元素超差和NaN。见[入口适配](../issues/REF-sfdp-pattern-15-native/性能入口适配分析.md)及[数值失败](../issues/REF-sfdp-pattern-15-native/性能OFF数值失败.md)。
19号额外half微图与FP32 mask组合编译失败；恢复原社区FP32域后独立OFF/ON及六臂计时已完成，PERF_REGRESSED。
ON仍是数学展开，half失败和原社区无数值oracle的限制继续保留，见[19号讲解](../results/current/T-105/pattern-19_讲解.md)。
18号六臂已完成，`PERF_IMPROVED`，见[功能性能讲解](../results/current/T-105/pattern-18_讲解.md)。
23/24六臂也已完成，分别PERF_IMPROVED/PERF_REGRESSED；执行原gate绑定的worker/helper快照，不被后续公共worker修正替换。
20号OFF出现真实NaN数值失败，21号OFF被22号接替；均阻止计时，不当作“性能免测”或通过。

## 1. 为什么不能所有编号都用 inference 图

冻结 PyTorch commit 为 `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。注册名称的编号相同，不保证输入图仍能由该编号独立匹配：

```python
# torch/_inductor/fx_passes/fuse_attention.py:1485
if "dropout_p" in workaround:
    pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
    replacement = partialize_and_update_signature(replacement, dropout_p=0.0)
# 同文件:1492，inference注册还设置skip_duplicates=True。
inference_name = name + "_inference"
```

例如 pattern 3 的本意是 `softmax(QKᵀ / scale) -> dropout -> matmul(V)`。dropout=0 后，它可能成为 pattern 1 的无 dropout 图。若以这种图计时，即使总 `fuse_attention` 计数增加，也不能归为 pattern 3 收益。

## 2. 当前派生性能方案

| 编号 | 准备的执行阶段 | 数值和归因条件 |
|---|---|---|
| 1/2/5 | half training，保留requires_grad；仅前向，不含backward | 与本轮已部署注册修复同阶段；不删除邻接制造OFF，不外推推理收益 |
| 3/4/6/7/9/12/28 | half training 注册，保留 requires_grad；仅测前向；dropout=1e-11 | 原社区完整合同先验证；新性能图单独比较 eager；必须本编号精确命中 |
| 19 | 同编号 FP32 inference，与原社区dtype域一致 | 额外half+FP32 mask编译失败原件保留；新图仍须独立门禁 |
| 其他已确认编号 | 同编号 half inference | 独立 OFF/ON、数值、精确编号与实际生成代码复核 |
| 16/29 | 暂不执行性能 | GPU 本编号合同仍未确认；不借用邻接编号 |
| 17 | 不做独立性能 | 静态等价去重别名，归并15，历史记录保留 |

极低非零 dropout 的设计来自冻结社区测试，而不是任意放宽容差：

```python
# test/inductor/test_fused_attention.py: _test_sdpa_rewriter_7 / _test_sdpa_rewriter_9
attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
self._check_common(..., has_dropout=True, override_check_equal=True, atol=2e-3)
# test/test_transformers.py:3853，社区也用该极低非零概率与dropout=0做数值对照。
dropout_p = 0.00000000001
```

它的作用是尽量避免随机 mask 差异干扰逐元素比较，同时在语义上保留非零 dropout。**不保证任意编号一定保留改写**：若常量折叠、guard 或注册去重使目标消失，功能门禁必须失败，不能改用邻接计数。若真实发生随机丢弃导致比较失败，保留失败，不重抽种子或改容差以制造通过。

这是社区方法启发的**派生 microbenchmark**：输入仍来自 `_get_sfdp_patterns` 的 shape/stride/dtype；初始化固定种子，张量不是未初始化的 `empty` 数值。它不是原社区性能 benchmark，不覆盖常见 dropout=0.1/0.5 的完整随机性正确性，也不代表完整训练、backward 或模型端到端收益。

## 3. 具体实现和门禁

```python
# runners/t102_t107_attention_performance_worker.py: select_registration
suffix = '_training' if training else '_inference'
eligible = [item for item in candidates if item[0].endswith(suffix)]
if pattern in LOW_DROPOUT_TRAINING_PATTERNS:
    workaround['dropout_p'] = 1e-11
# 原注册dict不修改；没有所需阶段时拒绝替换为其他阶段。
```

两臂保留整轮 joint graph，只删除 OFF 的精确编号注册；如果其他 attention 接替 OFF，也拒绝目标级收益归因。benchmark 入口要求 GPU、完整社区 NPU、ON 功能与 OFF 功能四类原件哈希和人工签发 gate。训练候选尚未部署的编号不得把隔离候选结果冒充安装态功能 gate。

计时前还校验原社区、OFF、ON为三个独立PID，Python解释器一致，实际加载源码逐文件哈希与当前安装态一致；
OFF/ON冻结源码必须clean、后端必须导入前选择。缺少源码指纹或功能验证后产品文件变化时，旧gate失效，先重验功能。
这些是计时入口校验，不会修改已经归档的原功能结果。

开始首次功能臂前进一步复核了mask数值域：15/20的search判断`mask==0`，replacement判断`mask==1`，
若用任意正态浮点初始化二者并不等价，不能把这种派生输入错误当产品缺陷。
当前15/20使用社区`ones(...).tril()`的0/1设计；18/19布尔mask也取下三角，保证至少一列可见。
shape/stride/dtype仍来自注册输入，18的广播mask不声称等同原社区完整方阵causal覆盖；
21/22/24的加性浮点mask仍保留浮点数值域，不擅改成bool以开启融合kernel。
上述初始化修正在首次功能运行前完成；此句记录时序，不表示目前仍未执行。

每臂启动即保存worker、OFF控制器、精确目标观察器三份源码快照及哈希。
新增`scripts/review_attention_functional_gate.py`：人工读取本编号FX/IR/codegen后才填写复核说明，
入口验证完整社区安装态合同、独立OFF/ON及当前加载源码，归档文本并签发门禁；它本身不运行设备或计时。

### 2026-09-14 23:11 CST：首次实际预检及入口适配

8个已通过原社区合同的编号，直接对内部`search_fn`调用compile均在FX捕获前触发`MOD_SKIPLIST`。
原失败日志、命令与源码快照分别归档在各issue的`evidence/functional-20260914T2303*`至`T2304*`目录，
`inventory.json.failed_precheck=true`。它们不能记为目标不支持或性能退化。

```python
# runners/t102_t107_attention_performance_worker.py
traceable_search = torch._dynamo.dont_skip_tracing(search_fn)
model = functools.partial(traceable_search, **scalar_workaround)
compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
# 冻结 torch/_dynamo/decorators.py:1611提供局部包装器；
# test/dynamo/test_decorators.py:1958包含递归跳过函数的对应社区验证。
```

只对当前测量函数局部允许追踪，未修改全局skip列表、pattern guard、替换逻辑或数值容差。
修正后先跑13号独立OFF/ON：OFF保留132项其他注册，总attention计数0；ON本编号改写1次。
两臂同FP16 `[1024,128,128]` 输入对齐eager，生成代码分别为bmm-softmax-bmm与NPU fusion attention。
对应性能gate及六臂现已完成；其他编号不能借用13号入口验证结果或数值结论。

静态回归检查覆盖阶段选择、原参数不被原地修改、dropout 参数缺失拒绝和命名一致性。
下一步仍按编号分别处理，不能把13/18的计时完成推广到全部attention。

## 4. 原社区功能PASS究竟比较了什么

原例通过、精确目标命中、数值通过是三个不同维度，不能由前两个自动推出第三个。

```python
# test/inductor/test_fused_attention.py:137，_check_common
if not has_dropout or override_check_equal:
    self.assertEqual(result1, result2, atol=atol, rtol=rtol)
# 同文件:140，training时实际执行两侧backward；下面条件满足才比较输入梯度。
if training:
    result1.sum().backward()
    result2.sum().backward()
    # 对每个浮点输入：
    if not has_dropout or override_check_equal:
        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=rtol)
```

冻结源码各原方法的实际参数复核如下。此表解释**源码断言范围**，不是宣称全部NPU方法已经执行通过。

| 编号 | 原社区输出/输入梯度比较范围 |
|---|---|
| 3/4/6/12/28 | `has_dropout=True`且不覆盖默认开关：不比较输出/输入梯度；仍执行训练/推理和backward及匹配断言 |
| 19/20 | 同样不比较输出/输入梯度，且`check_train=False`，只执行推理和结构断言 |
| 7/9 | 极低非零dropout、`override_check_equal=True`，比较输出和训练分支的输入梯度 |
| 13 | `override_check_equal=True`但`check_train=False`，只比较输出 |
| 15/18/21/22/23/24 | 只推理，比较输出，不覆盖输入梯度 |
| 1/2/5/8/10/11/14/30 | 原方法覆盖的训练分支比较输出及浮点输入梯度；具体dtype/shape仍以本方法为准 |
| 16/17/29 | GPU本编号归因未完成，不能拿已有数值/结构通过替代；17原例也没有数值比较 |

因此已确认GPU编号的24个原例中，有**7例（3/4/6/12/19/20/28）没有输出/输入梯度数值比较**。
NPU若通过这些原例，只能写“原社区结构/执行合同通过”，数值oracle缺口必须单列。
性能派生图的独立eager比较可以支持那张微图的计时门禁，但不能回填原高dropout合同为精度已验证，
也不能将“训练前向时延改善”写成完整前向+反向训练收益。
