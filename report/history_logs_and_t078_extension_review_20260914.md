# 历史日志补齐与 T-078 邻接闭环

> 更新时间：2026-09-14 22:20 CST（UTC+08:00）。收件仓库HEAD `16ecd84`；不执行GPU、不改写历史原件。

## 本轮已关闭什么

| 任务 | 新收件 | 实际复核 | 本轮关闭的缺口 |
|---|---|---|---|
| T-076 | `results/incoming/T-076/history-logs.json` | 13份非空stderr，绑定2026-09-01原run及原inventory SHA | 13/13原始测试日志重解析 |
| T-077 | `results/incoming/T-077/history-logs/manifest.json`及3片 | 11份非空stderr，绑定2026-09-02原run及原inventory SHA | 11/11原始测试日志重解析 |
| T-078 | `results/incoming/T-078/FP16-value1-text-handoff.json` | 1/1 FP16、value=1派生例；15份正文、原始FX/IR/codegen | 已登记FP16 value=1 GPU邻接缺口 |

T-076/T-077 24份空stdout由原inventory的零字节和空文件SHA确定，不必再上传。24份非空日志均已验证，
**不要重复让GPU操作者导出这些日志**。T-078本次确为正确case；旧误命名codegen邻接仍保留，不覆盖。

## 为什么严格历史总状态还是 pending=41、exempt=3

严格状态是24个GPU case、10个NPU单元、7个性能处置的综合门禁，不是41个文件，也不是本轮没有进展。
24个日志子检查已全部通过，但综合门禁仍有：

- 全量archive未齐：review正文/日志可用，不代表旧inventory登记的所有debug/structured trace原件齐备。
- 2个GPU数值辅助oracle边界：`pad-mm-original-aten`仅检查counter/生成符号；`pad-mm-exclusion`仅检查cache排除。
- 3个GPU梯度oracle边界：decompose-bmm、decompose-mm-fp32、decompose-mm-mixed的无参数module梯度比较为空字典，不能证明输入梯度一致。
- 10个历史NPU单元缺少充分的运行态源码/补丁、三库revision和导入前后端绑定；19份已选择原结果可读不等于这些元数据已存在。
- 历史性能缺逐样本或逐worker完整溯源；4项汇总复算一致，但不能逆推出未记录的原样本、PID或执行时源码。

新增数值/梯度断言需按**新run**记录，不能补写到历史测试中。GPU重跑也不能生成旧NPU运行的缺失溯源。
显式关闭的3项仍免测，不为凑ON性能绕过产品disable。旧结果原样保留，不把其他后端历史搬入experimental。

## T-078 对齐边界

登记variants从5已核验+1待补更新为6已核验。原4个单元分母不变，项目comparison/性能处置仍各42。
GPU/NPU都为div+add、FMA counter=0，各自compiled与本设备eager位级一致；NPU quotient需要显式FP16舍入，
CUDA本例不含该中间舍入，**仍为PARTIAL_ALIGNED**。这不是新增FMA收益，value=1仍不进FMA ON性能分母。

详细测例来源、源码块、调用链、两端实际生成代码和复核命令见
[GPU补证与NPU修复对照](../issues/REF-addcdiv-fma-fp16-value1-derived/GPU补证与NPU修复对照.md)。

## 持续复核入口

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_t078_value_one.py \
  --check-current
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_all.py \
  --write-audit
```

零设备工具门禁通过不等于历史再认证通过。需要强制后者时追加 `--require-history-complete`，
现有缺口下应返回3而非PASS。固定机器入口：[最新历史审计](../results/audits/latest.json)。
