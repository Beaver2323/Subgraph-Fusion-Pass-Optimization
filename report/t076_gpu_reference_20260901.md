# T-076 GPU 原生 Reference 验证报告

> 报告整理时间：2026-09-02 02:22 CST（UTC+08:00）
> GPU run：`reference-20260901T180826+0800`
> 结论：`valid-reference-suite`，13/13 passed，13/13 `reference_valid=true`

## 1. 验收结论

GPU 机器直接运行冻结 PyTorch commit 中的 13 个 community tests，没有 adapter/extracted case。
文本 handoff 的 suite、case audit、环境与关键文件哈希互相一致：

| 项目 | 结果 |
| --- | --- |
| acceptance units | 5 |
| manifest variants | 20 |
| direct cases | 13/13 passed |
| reference valid | 13/13 |
| 动态 reference variants | 14/14 `valid-reference` |
| static registration variants | 3/3 `static-registration-evidence-only` |
| NPU-only guards | 3/3 `not-applicable-reference-guard` |
| failed / skipped / timeout / no-tests | 0 / 0 / 0 / 0 |
| case 总耗时 | 202.889325 s |
| inventory 聚合字节数 | 5,800,343 bytes |
| adapter 决策 | 13/13 `not-needed-direct-valid` |

因此首批 5 个 acceptance units 从 `mapped-static-await-gpu-reference` 冻结为 `frozen`，进入
`yes-frozen` denominator。正式闭环数仍为 0，必须完成当前 Pass 环境的 NPU comparison 才能关闭。

## 2. 环境证据

| 项目 | 实际值 |
| --- | --- |
| 环境生成时间 | `2026-09-01T18:08:36+08:00` |
| 主机 / 用户 | `node-5-15` / `z00824525`，UID/EUID 1039 |
| 工作目录 | `/data/z50063656/tmp` |
| GPU | 8 × NVIDIA A100-SXM4-80GB，SM 8.0，80 GiB |
| 宿主驱动 | `550.54.15`；Driver API `12040` |
| CUDA Toolkit | 12.6，nvcc `V12.6.85` |
| CUDA compat | 未启用，`CUDA_COMPAT_DIR=null` |
| Python | 3.12.13，`/data/z50063656/envs/PassGPURef/bin/python` |
| PyTorch | `2.14.0a0+git8e86e0a`，CUDA 12.6 |
| PyTorch commit | `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| PyTorch source | `/data/z50063656/src/pytorch`，working tree clean |
| cuDNN | 92501（9.25.1） |
| Triton | `3.8.0+git675c5987` |
| GPU runtime gate | `cuda_available=true`、device count 8、`has_gpu_and_triton=true` |
| environment fingerprint | `8f0b3a0d3291d9c70872444dc3c430e12a7593aeaa1a5e1576a93da62161d7ee` |

`CONDA_PREFIX=null`，实际解释器位于 pip venv。`PATH` 中仍可见 Miniconda `condabin`，但解释器、
PyTorch source、CUDA、cuDNN 和执行工作目录均由环境指纹明确固定，不构成本轮 blocker。

## 3. Case 审核

所有 case 都满足：return code 0、tests ran 1、tests skipped 0、before/after FX captured、冻结 commit
一致且 adapter 不需要。

| Case | Unit | 耗时(s) | FX before/after | Artifact bytes |
| --- | --- | ---: | ---: | ---: |
| `REF-mm-plus-mm-native` | `AU-post-grad-mm-plus-mm` | 13.631077 | 4 / 4 | 662991 |
| `REF-pad-mm-dynamic-m-native` | `AU-pad-mm-mm` | 12.381609 | 1 / 1 | 288352 |
| `REF-pad-mm-original-aten-native` | `AU-pad-mm-mm` | 13.432121 | 1 / 1 | 224188 |
| `REF-pad-mm-stride-native` | `AU-pad-mm-mm` | 14.034355 | 1 / 1 | 171697 |
| `REF-pad-mm-exclusion-native` | `AU-pad-mm-mm` | 11.633685 | 2 / 2 | 305331 |
| `REF-pad-bmm-dynamic-batch-native` | `AU-pad-mm-bmm` | 17.943073 | 1 / 1 | 306276 |
| `REF-pad-bmm-static-fp16-native` | `AU-pad-mm-bmm` | 17.543099 | 1 / 1 | 203608 |
| `REF-pad-bmm-autocast-regression-native` | `AU-pad-mm-bmm` | 22.751447 | 2 / 2 | 932076 |
| `REF-pad-addmm-dynamic-m-native` | `AU-pad-mm-addmm` | 19.295635 | 1 / 1 | 291228 |
| `REF-pad-addmm-bias-native` | `AU-pad-mm-addmm` | 14.588221 | 6 / 6 | 1168008 |
| `REF-pad-addmm-beta-zero-negative-native` | `AU-pad-mm-addmm` | 15.887572 | 1 / 1 | 147686 |
| `REF-addmm-contract-native` | `AU-post-grad-addmm` | 16.636087 | 5 / 5 | 874488 |
| `REF-addmm-symbolic-scalar-negative-native` | `AU-post-grad-addmm` | 13.131344 | 1 / 1 | 224414 |

## 4. 逐 Case 稳定签名与结构化结果哈希

| Case | FX stable signature | reference result SHA256 | inventory SHA256 |
| --- | --- | --- | --- |
| `REF-mm-plus-mm-native` | `df1c3b19f907f9144c3cba2bf6dbe723ac926c7ea81644e3f3215fc3f02c503b` | `93f2d14f28fbe55a3950f09d02867632dcee4710b4c3ccb15635a785fbf00953` | `703c706c7b1528d5091e1310ed71cf896609dfc09dabc3ec164256d2b979a1c3` |
| `REF-pad-mm-dynamic-m-native` | `10f3605208f43be7d01e71342016c3cf7bd60cf9e473f153eeff5fa8f95f8fd1` | `7662c544b807661fe4aa9a8525acb6f8f591de7911c9a9dec0619ae600015e28` | `e2ff3783dc2d9cd092633c0ccb3e27c64e68f0aa0df908c419826fab85ae1520` |
| `REF-pad-mm-original-aten-native` | `0f5494422d73b1c2f7faac5c1888d1af686ce1dc045396819af126dd33795472` | `b0db7cb67bfc8bd8cc2d3d256f8022c345d0a978c22a8d5938d7bb180ed55a5b` | `190dd7330e89386ce6551a2281cce7b2be2f68996040dfc7c526085c0b5dee7c` |
| `REF-pad-mm-stride-native` | `a68d7527bc8f980ee7ea10cf4eafbb6e9c6a3f9c2c6c6b3784cc272b9cebbea3` | `4dbcf21b0832219b3fc4cff5f35213335ef9be8781a8ec065356fbe353452d07` | `76a63886320e9d7e3ed60f665ef4c6dd33530b6cb39d15513bf21e9454437230` |
| `REF-pad-mm-exclusion-native` | `f0c640eab755ac39be8bb7377ba0668cb6517139cce72273af6ef7333bad0673` | `c9d918c101818b2d1c9ad401594f29c493509f7aa96143499b90b8f5bf1589c9` | `e512221858bb5f1f14becc072015030b831ff2326293b3e3d2548d7c5404f23a` |
| `REF-pad-bmm-dynamic-batch-native` | `67270f3578f38757e72cdfec6e864ac6959dbbd29b30097e52654cf21563626b` | `2f8932baf9dd1f74de5904ed7e565c512e40c72d902a448f49d1452cf1208e3c` | `512b6963a71d89226c7234ad0e06cecaaaca6ca06fe1ac6c86d89f861a7ff353` |
| `REF-pad-bmm-static-fp16-native` | `c02126ce986d13ed50c2d2dcb75cf2ea4f523ff4a2d9283adad36eb2cc334e25` | `9de1b3a5ff36c5fe0391ae305beb5b3202a62c0ba3719bcd076436f1d800f3dd` | `7a2749149ef6218477372727898c9d00cdea0cf6b0f6d17d4c9d271d5e5b5856` |
| `REF-pad-bmm-autocast-regression-native` | `6df7cd57a6b917ee6420190e2e80a9e5abbf036985e242af5b4a6c217add03be` | `6e8609295ea1fd7268e5617332ad8325af36659d5f0a1f2ab943e1d415da2af5` | `85506689a995beb58ce7b356454c765580452c13c5a88fe5ddf6afcb16f301d4` |
| `REF-pad-addmm-dynamic-m-native` | `1ab0f6f8c206fd7260a434d53f9525a90ea7ded638b59f37ff4e589bb3dcafed` | `3939e1ee8d3ce951358f002f2974a1dda9b12fc9b48224b04249a84bbb906a74` | `2e469df4f864ed5547e235e425f811be7195011b03e6c99dbb1eb2295ce760fd` |
| `REF-pad-addmm-bias-native` | `769aa214cbf847f308dfa6ac390e9b7a6b7006f42d6026959fcb88954fc83347` | `1fd45e566cb68114e1a5096ed3393592c98fd17066763d84cc268629823c5096` | `af2d425a985998487e1855abb3b8bec9780a236b0d20fd0edae6a03d65de0227` |
| `REF-pad-addmm-beta-zero-negative-native` | `42a83718bfbf0a6f6e7a34774a527d3254745acd806a97555876ef2dfbed6b01` | `28ae1626290d0775fe3c0c27c8ae82c306bbad0d210e5fa942543350f4849e93` | `c8c8bae7d1198a76bb69d67b82cb27bdec093eb072722f4dc760466fec39edb6` |
| `REF-addmm-contract-native` | `a367f9154e7b12211fff4baf966a0e8cb8ddb99e6967cb3c9d39cf8eaa80727f` | `eb1dd1e312082e385ebf418031662a6db4d3b8723f66e4cf7e74df40383b7b41` | `38e21d1bc5c0387f8322afaa7358f4007b917d49144c25abe21ffdb1e8744336` |
| `REF-addmm-symbolic-scalar-negative-native` | `9a454d6d4affc275d2884e09f32da4c27c706976e1366579d57f0ecdc1a95bf2` | `c2b53ab05a577ff5f506ceacf99c8ebf1a8a712edee0d1480cddf0619a1bb35b` | `7851f299e40e53a4d6ff5711ed4dbd69403c6ea99db67feb21d73c9c0ac4b698` |

## 5. Run 级哈希与保存边界

| 文件/对象 | SHA256 |
| --- | --- |
| environment JSON | `e593021403b8650b32a1da967936a606f71430ab36082922b33b93cd6fc67e7b` |
| manifest snapshot | `fd68168e7f0b832509eeaf6c756ece096b96b16f17c2d304ec73f2ce51c19d63` |
| reference plan snapshot | `6029cfd5d11a2f19259c11bfabc240299e574d9ffb069173a6248c70b484d73d` |
| reference summary | `21d57950fbe197ce6460d284d8b77e329deb7714f7128adcbdad6fb4d20855e0` |
| handoff canonical payload | `7fb9033a765dd632fc3288640ac33803a30a9d63848acddb14f48065b11b6a8d` |

GPU 机器禁用 Git/二进制上传，因此仓库保存可审计摘要和完整结构化哈希，不复制大体积 debug
正文。原始 run 保留在 GPU 路径：
`/data/z50063656/tmp/t076-reference-results/reference-20260901T180826+0800`。若后续需要检查某个
具体 FX/codegen 文件，必须在 GPU 机器按本报告的 inventory/result hash 回查，不能从摘要臆测正文。

## 6. 后续动作

1. 不创建 GPU adapter；direct 路径已经全部有效。
2. 首批 denominator 冻结为 5 个 acceptance units，正式闭环仍为 0/5。
3. 从同一 manifest 生成 NPU runner，使用当前 Pass 环境和 `triton_experimental` backend。
4. NPU 侧分别保留产品默认 gate 与仅用于诊断的 gate-bypass；不能把 bypass 冒充 baseline。
5. 完成 execution、correctness、FX、runtime path 和 first divergence 后生成 5 个 comparison verdict。
