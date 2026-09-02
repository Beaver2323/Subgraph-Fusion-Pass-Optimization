# T-077 GPU 原生 Reference 验证报告

> 报告整理时间：2026-09-02 21:03 CST（UTC+08:00）
> GPU run：`reference-20260902T125636+0800`
> 结论：`valid-reference-suite`，11/11 cases passed，17/17 variants 为有效 reference

## 1. 验收结论

GPU 机器在冻结 PyTorch commit 上直接运行 11 个 community cases，没有 adapter 或 extracted case。
其中参数化 decompose 方法实际执行 23 个 unittest 实例。文本 handoff 满足以下门禁：

| 项目 | 结果 |
| --- | --- |
| acceptance units | 5 |
| direct cases | 11/11 passed |
| reference valid | 11/11 |
| variants | 17/17 `valid-reference` |
| failed / skipped / timeout / no-tests | 0 / 0 / 0 / 0 |
| case 总耗时 | 198.815200 s |
| artifact 聚合字节数 | 7,359,380 bytes |
| adapter 决策 | 11/11 `not-needed-direct-valid` |

因此 T-077 的 5 个 acceptance units 从 `pending-reference` 冻结为 `yes-frozen`。冻结只表示 GPU
reference 合同有效；必须完成 NPU execution、comparison 和 first-divergence 分类后才能正式关闭。

## 2. 环境证据

| 项目 | 实际值 |
| --- | --- |
| 环境生成时间 | `2026-09-02T12:56:45+08:00` |
| 主机 / 用户 | `node-5-15` / `z00824525`，UID/EUID 1039 |
| 工作目录 | `/data/z50063656/tmp` |
| GPU | NVIDIA A100-SXM4-80GB，SM 8.0，80 GiB；本轮 `CUDA_VISIBLE_DEVICES=2`，进程内可见 1 张 |
| 宿主物理设备 | 8 × NVIDIA A100-SXM4-80GB |
| 驱动 / Driver API | `550.54.15` / `12040` |
| CUDA Toolkit | 12.6，nvcc `V12.6.85` |
| Python | 3.12.13，`/data/z50063656/envs/PassGPURef/bin/python` |
| PyTorch | `2.14.0a0+git8e86e0a`，CUDA 12.6 |
| PyTorch commit | `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| PyTorch source | `/data/z50063656/src/pytorch`，working tree clean |
| cuDNN / Triton | 92501 / `3.8.0+git675c5987` |
| environment fingerprint | `744891fb4f6678126e652b20d7e1e0635d568c7c2dafc2b749474a5902e941ef` |

## 3. 逐 case 证据

| Case | Unit | tests | FX before/after | FX stable signature | reference SHA256 | inventory SHA256 |
| --- | --- | ---: | ---: | --- | --- | --- |
| `REF-gumbel-max-trick-native` | `AU-apply-gumbel-max-trick` | 1 | 1/1 | `0368d9c3bdf88bf40a1f1300fcd3d589d85bd3c68316710247cfd3630d17b995` | `95a53400b6d387a235b817d00dfeedfbee37edb228dc122232177311b9179e5a` | `3eb75fe02095522a83a37908d636dc68bdf8bdfe0ebf4f95bab661957a7f75cf` |
| `REF-b2b-gemm-left-gelu-native` | `AU-b2b-gemm` | 1 | 1/1 | `6c2ed09ddf2f822ca5b67aaa45b3f856f9adff2db46ab463a4785a974ef4f627` | `a947f92855adfc1f0a13ea02a29867410356ae8641f6928971202122f21d1c32` | `4c2ea2657a94350fa43099686a65549f9807b1b52c33cd7910f88f59acfe293d` |
| `REF-b2b-gemm-right-relu-native` | `AU-b2b-gemm` | 1 | 1/1 | `fa621f172a9bf91c1051aa7659cba43509038f9b06282cc3c024fd4f50247c05` | `8514152bf9320fe35c88d7ca7f10b71a679eebc2861f7cec97df9da309bdb65a` | `44f1e80afe81734f8a77ebf22ba618a9723de072005cc647c1505d248824fca7` |
| `REF-b2b-gemm-trivial-left-native` | `AU-b2b-gemm` | 1 | 1/1 | `b29b414fff50083b4647997e338980bd2f63b4f6d6938d3219d38e61ae56b658` | `376a431d64d27070be0df2f451b5ac37ff2473c9bb1dd5cb495eb9fa0153df68` | `f2dbfdd8b073eb5824e55dd048a6da0421e1b97654cda37d659fa08a0662f480` |
| `REF-b2b-gemm-trivial-right-native` | `AU-b2b-gemm` | 1 | 1/1 | `af601c6e173d6fdf60204bca6bd9876772a060bce8fdc45d2ed3e054b0d2defb` | `865c235e579da8c5c4c2f594696d8a5062ce73a7f3d16acae9dd44d8ae515880` | `63871e79b658b0730aa9c5390150b6b3f5d3909f6fe4591a7e9ed0278aa71c69` |
| `REF-b2b-gemm-bad-pattern-native` | `AU-b2b-gemm` | 1 | 1/1 | `95b21e723e99614e6313934ba832dc87d1dc875afdc05bec19b46808d4f77880` | `3330c38bfbf1f965c468607eca903ee489d24d1a4569d9c276a2d8cdcd18685f` | `29d195e24ef662cc1c7b0d86019cbffcc121efca3bd6dfa24d01ea823d569fa6` |
| `REF-b2b-gemm-bad-shape-native` | `AU-b2b-gemm` | 1 | 1/1 | `947d2753a2d31206c9a99b9cded03aca2dbc6f163084ba9cff5f5d4960b0e27c` | `544844b019b11b5d43e6e6b26fd2eb039681a21c63d0e0ab742e3f53c88ae14c` | `a68f8deddf8101bd79d35d871e26a53e4bd4995364895317785866ea4f61eb25` |
| `REF-decompose-bmm-native` | `AU-decompose-mem-bound-mm-decompose-bmm` | 3 | 6/6 | `fa2f379859ba437743fbad484a44e5db63cf5d5668c8b2323e0e13f1987950d1` | `06e6514c4519d3b51299fcb425e618648adb6dd38597eda8b48626b5e47cf702` | `78d103ab6802d54d9901b722bc3228cd26a5f18cef7b89b66ccb52b0530feb77` |
| `REF-decompose-mm-fp32-native` | `AU-decompose-mem-bound-mm-decompose-mm` | 6 | 12/12 | `9f7ae82dfc1ebac2932a0950e4ea82ece95fb592ad86bcde1601acd440bcf1a2` | `140d6d193c62046d5f24692a8f2900981c03c466665c7918d0fc21ad6b5c2893` | `261fa06f1bec40a4048ef92e19b203fa2054fb6dbeaed87afb4ae39935eb14b2` |
| `REF-decompose-mm-mixed-native` | `AU-decompose-mem-bound-mm-decompose-mm` | 6 | 12/12 | `77059193d92e823b5aa35b36569b8ebae912fd7a5915c5a15c6d20fe8f185bd6` | `618c10f8469a5fdd68829ab6f1fb13c3c9effd58f3252b8246fe6753d31d707b` | `724d49785398d3de87df862cea500b805806a2a0c84e1705feab9c8e900b67b7` |
| `REF-decompose-addmm-dynamic-native` | `AU-decompose-mem-bound-mm-decompose-addmm` | 1 | 1/1 | `b5a5ccfbac653e996bb408d2fcefda09bc5acf8e80e6f8c2a96b6a946e3ce1c6` | `e8e1f96379e07b23f675ff61f13f28e64685a982ef5ddea78794cc80a5e99100` | `e2c2101f640e64cd68f849a139714a27425d7f3955c0af8d2003534a25c39782` |

## 4. Run 级哈希与保存边界

| 文件/对象 | SHA256 |
| --- | --- |
| environment JSON | `dc4c9657f4c2c79682cef831b52230eadd2759277e3e5d710085665e0872141f` |
| manifest snapshot | `659a9778d4a1db92f9ef66148eedeef973a22cf675405e219c5af969d6dd1408` |
| reference plan snapshot | `f93bbc906d935faeb6100267cdb31314e9530959fff30ccd40e3b9d4060b6d17` |
| reference summary | `880c0411478bcf6427035e840059d955bb31e09c2c3d36861f0c6e0323c1307a` |
| handoff canonical payload | `97a1ef8315c9d1332dcee4542c928aedc600ebc984b2ac3e4e5ab27e0079b518` |

GPU 机器禁用 Git/二进制上传，因此仓库只保存可审计摘要和结构化哈希。原始 run 保留在
`/data/z50063656/tmp/t077-reference-results/reference-20260902T125636+0800`。需要查看具体 FX、
generated code 或 stderr 时，必须按本报告哈希回查原始 GPU 目录，不能从摘要推断正文。

## 5. 后续动作

1. 不创建 GPU adapter；11 个 direct cases 已全部有效。
2. 第二波 denominator 冻结为 5 个 acceptance units、17 个 variants。
3. NPU 侧仍按原生入口优先；原生 test body 被 GPU_TYPE/`requires_gpu` 阻断后，才运行最小 adapter。
4. 每个 variant 必须记录源码意图、GPU/NPU 行为、FX/replacement/codegen、正确性和 first divergence。
5. 只有 `NPU_REGRESSION` 进入 repair；平台适用性或显式产品控制不强行修复。
