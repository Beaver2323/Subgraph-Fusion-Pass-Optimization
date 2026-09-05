# T-076 / T-077 历史证据复核与统一门禁

> 更新时间：2026-09-06 03:02 CST（UTC+08:00）
> 规则版本：`2026-09-06.1`；校验器版本：`2.0.0`。
> 范围：历史证据只读复核、检查工具修复；没有新跑 GPU/NPU，没有修改产品代码，没有提交推送。

## 1. 当前结论

T-076/T-077 的 **10 份 NPU/comparison 记录、37 个 variants** 已通过当前记录校验。
这只证明结构、后端声明、结果哈希与 verdict 逻辑符合新约束，不等于历史运行已全部重新认证。

原始 `results/current/`、历史报告和原始运行文件保持不变；复核结果追加在
[`results/audits/latest.json`](../results/audits/latest.json)，该固定入口指向带时间戳的独立审计文件。
每次重新生成都会保存规则版本、校验器版本、实际代码/计划哈希、仓库 HEAD、已读取源文件哈希和时间戳。
HEAD 不能代表未提交代码，因此还单独保存 `validator_fingerprint_sha256` 和逐文件哈希。

| 任务 / 单元 | GPU 原始日志重解析 | NPU 记录检查 / 原始运行复核 | 性能新规则复核 |
| --- | --- | --- | --- |
| T-076 mm_plus_mm | 1 case 待补原文件 | 通过 / 待复核 | 历史复用合同待人工核对 |
| T-076 pad-mm | 4 cases 待补原文件 | 通过 / 待复核 | 保留显式关闭免测 |
| T-076 pad-bmm | 3 cases 待补原文件 | 通过 / 待复核 | 保留显式关闭免测 |
| T-076 pad-addmm | 3 cases 待补原文件 | 通过 / 待复核 | 保留显式关闭免测 |
| T-076 post-grad addmm | 2 cases 待补原文件 | 通过 / 待复核 | P-018 候选与当前安装态的复用边界待核对 |
| T-077 Gumbel-max | 1 case 待补原文件 | 通过 / 待复核 | 原始六轮可读；逐轮后端/revision/生命周期绑定不完整 |
| T-077 B2B GEMM | 6 cases 待补原文件 | 通过 / 待复核 | 社区网格、候选审批、实际模板选择证据待人工复核 |
| T-077 decompose-BMM | 1 case 待补原文件 | 通过 / 待复核 | 原始六轮可读；完整 revision/生命周期与测量方法待核对 |
| T-077 decompose-MM | 2 cases 待补原文件 | 通过 / 待复核 | 原始六轮可读；候选、完整 revision/生命周期待核对 |
| T-077 decompose-addmm | 1 case 待补原文件 | 通过 / 待复核 | 原始六轮可读；完整 revision/生命周期与测量方法待核对 |

审计总状态为 `pending`：24 个 GPU case、10 个 NPU 原始运行复核项、7 个性能处置复核项待完成，
另有 3 个性能免测项。**41 是复核项数，不是 41 个 pattern，也不是新增任务分母。**
10 项 NPU 记录检查通过另行计数，不抵扣上述原始证据缺口。

Gumbel 的历史收益数值没有被撤销或重算；本轮增加的是“能否按更强证据约束重新认证”的状态。
不能把“原报告写了后端”当作每个 worker 已采集导入前后端选择和精确 revision 的证明。
T-077 MM 的 `dfbcc25` 仍为已验证、未合入候选。本次不会自动升级其产品合入状态。
T-067 属于旧 autotune feature-family 格式，不在本次 T-076/T-077 的自动迁移范围内。

## 2. 已落实的约束

- [`audit_policy.json`](../schemas/audit_policy.json) 集中登记规则版本与边界；规则变化必须更新版本并重新生成复核记录。
- 当前 reference parser 拒绝部分 skip、expected failure、unexpected success、数量不符和非完整 OK；代码生成断言不能直接充当数值正确性。
- 当前 NPU/comparison validator 强制 `triton_experimental`；失败证据允许保留，但不能伪装为修复完成或性能通过。
- 性能旧文件先检查已登记 SHA256，再核对六轮唯一编号、输入/迭代数、正确性与 OFF=0/ON=1 target counter；缺失完整溯源时保持 `pending`。
- 明确关闭项只核对免测记录与对应 `npu_control`，不额外运行 ON，不把 device guard 遗漏 NPU 自动解释为产品禁用。
- 历史摘要存在但原文件缺失为 `pending`；原文件与已登记哈希冲突、后端不符、功能失败等为 `failed`，普通模式也会阻断。
- 工具检查与历史再认证有不同退出码；所有检查只读当前证据。审计归档采用新建文件，拒绝覆盖已有清单。

本版历史工具是保守的复核清单，不是通用历史格式转换器：NPU 原始运行、断言语义、性能计时与
候选审批尚需逐项人工复核。即使补回 GPU 日志且重解析通过，也不自动消除这些待办。
后续完成项应新增有源文件/代码哈希支撑的复核记录并扩展相应检查，不能直接删掉 `pending` 来获得通过。

## 3. 统一检查命令

控制节点提交改动前执行；不占用 GPU/NPU、不导入 torch：

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_all.py --write-audit
```

默认检查代码语法、Shell 语法、JSON、零设备单测、记录交叉校验、T-078～T-080 计划、backlog、
全部五批 53 个 reference 入口及 whitespace。随后实时生成历史复核结果，不拿旧审计文件冒充本轮检查。
`--write-audit` 仅追加审计产物，并更新固定 `latest.json` 软链接；无需手工找时间戳。

若要宣称 T-076/T-077 已按新规则全部再认证，必须执行严格门禁：

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_all.py --require-history-complete
```

| 结果 | 退出码 | 允许宣称的范围 |
| --- | --- | --- |
| 工具检查通过，未要求历史完整 | 0 | 仅工具层通过；必须同时披露 `history_recertification` |
| 工具错误、证据冲突或实际复核失败 | 2 | 不通过 |
| 严格模式下历史证据待复核 | 3 | 工具可能通过，但历史再认证不通过 |

这是统一检查入口和工作流要求；本次**没有修改用户的 Git hooks 配置，也没有安装远端分支保护**。
不能把“有脚本”描述为服务器已经强制保护所有提交。

## 4. 补证据方式

原始 GPU run 保留在 GPU 机器，不要求立即重跑测试：

- T-076：`/data/z50063656/tmp/t076-reference-results/reference-20260901T180826+0800`；
- T-077：`/data/z50063656/tmp/t077-reference-results/reference-20260902T125636+0800`。

现有紧凑文本 handoff 只包含日志/FX 的哈希，没有原始正文，不能据此重解析。
在符合服务器传输限制的前提下补回原始文本文件后，可通过
`--gpu-run T-076=/本机原始run目录`、`--gpu-run T-077=/本机原始run目录` 指定位置；
工具不会联网、不会连接 GPU，也不会重新执行测例。它按原 comparison 登记的 result/inventory 哈希
绑定文件，再按 inventory 验证日志/FX 和其他已登记工件。
原始工件未齐全时仍保持待复核，不要求通过改变测试来补成“通过”。

## 5. 本轮额外修复与验证

GPU 一键入口使用 `run/text-handoff.json`，但原导出器禁止写入 run 内部，两者存在串联冲突。
现改为显式 `--allow-derived-output`，只允许新建保留文件 `text-handoff.json`；其他 run 内路径、
已有文件、软链接均拒绝。普通导出模式仍要求写在原始 run 外。
`latest` 发布成功与失败分支都拒绝软链接 handoff。

新增回归覆盖真实 CLI 导出→latest 发布、原始文件不变、覆盖拒绝、部分 skip 历史重解析、
错误后端、重复轮次、功能失败、缺证据、退出码分层与审计追加写。

本轮执行结果：44 个零设备单测通过，五批 53 个 reference cases 静态检查通过，
10 份 NPU/comparison、37 个 variants 记录校验通过。普通门禁退出 0，严格历史再认证门禁
实际退出 3（`pending=41`、`exempt=3`），符合缺证据不冒充通过的预期。
`results/audits/` 仅放行审计 JSON 与固定软链接入库，原始大日志仍按既有忽略规则留在运行目录。
