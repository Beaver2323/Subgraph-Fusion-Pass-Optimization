# Inductor Pass NPU 工作交接（2026-08-29，Conda Pass 环境，T-074 第一版）

> **2026-08-29 主线校准**：当前最高优先级需求见
> `/home/z50063656/Pass/需求变更.md`。主任务是评估 **PyTorch 社区原生 Inductor
> pass/pattern** 在 NPU `triton_experimental` 上的命中、可用性和性能，不是继续按
> experimental feature family 扩展。T-056 的 203 条 inherited-upstream 是候选库，
> 35 个 feature family 是验收组织单元；两者都不能直接当作已完成 pass 数。

## 1. 新对话先遵守的执行合同

1. 工作目录是 `/home/z50063656/Pass`。
2. 所有测试必须从 `/home/z50063656/tmp` 发起；不得在 PyTorch 或 torch_npu 源码树内 import
   torch。
3. 当前唯一主环境入口是：

   ```bash
   cd /home/z50063656/tmp
   source /home/z50063656/Pass/activate_pass.sh
   ```

4. 不要为新测试 source `/home/z50063656/Benchmark/env.sh`。T-055 至 T-073 中已经使用
   Benchmark 或独立 venv 得到的结果保留为历史证据，不能改写其真实环境。
5. 修改 PyTorch、torch_npu、Triton、runner 或安装 wheel 前，先在 `change_control.md` 新增条目。
6. 不清理共享 source tree，不终止其他用户进程，不覆盖用户/其他进程留下的无关 diff。
7. GitHub token 位于既有凭据文件；只有用户明确授权推送时才能读取，且不得输出 token。
   2026-08-29 用户已明确授权把本批文档更新提交并推送到文档仓；该授权不扩展到任何
   PyTorch、torch_npu 或 Triton 源码仓。

## 2. 当前主环境的只读验真结果

| 项目 | 当前值 |
|---|---|
| Conda | `Pass`，路径 `/home/z50063656/envs/Pass` |
| 激活脚本 | `/home/z50063656/Pass/activate_pass.sh` |
| Python | 3.11.15 |
| PyTorch | `2.14.0a0+git8e86e0a` |
| PyTorch import/source | `/home/z50063656/Pass/src/pytorch/torch/__init__.py`，`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0a0+git83cc452`，从 Pass site-packages 加载 |
| torch_npu source | `/home/z50063656/Pass/src/torch_npu`，`master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| Triton runtime | `triton.__version__ == 3.2.0` |
| Triton metadata | `triton 3.5.0`、`triton_ascend 3.2.2`，存在代码/metadata 错位 |
| Triton Ascend source | `/home/z50063656/Pass/src/triton-ascend`，`release/3.2.2@8bd9f380d2786002b84b5248f00838c26f900515` |
| CANN | 9.0.1 |
| NPU | 8 张 Ascend910B2，2026-08-28 核验时全部 Health OK |

已授权只读探针设置 `ASCEND_RT_VISIBLE_DEVICES=1` 后确认：torch_npu 可导入、
`torch.npu.is_available() == True`、可见设备数为 1、设备名为 Ascend910B2。普通受限沙箱不能
访问 HAL 时可能误报设备数为 0，不能据此判定环境失效。

### wheel 身份冲突：开始工作前必须处理

- Pass 环境 installed `direct_url.json` 记录的 wheel SHA256：
  `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`；
- `/home/z50063656/Pass/src/torch_npu/dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`
  当前 SHA256：
  `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989`。

文件名相同但内容不同。不要直接 `pip install --force-reinstall`。先决定 Pass 的 intended baseline，
记录回滚 artifact 和哈希，再遵守 `--no-deps` 安装规则。

## 3. Git 与工作树快照

### 文档仓库

- 路径：`/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization`；
- 本地 HEAD 与 `origin/main` 在交接前均为
  `a30979bc8f9fcdb9634b7983998dbe816499b39d`；
- 该提交为 `docs: archive experimental audits T059 through T061`，已推送；
- 上述基线之后累计的 P020 最终关闭追加、T062 至 T074、当前环境修正和本交接文件属于
  同一批持续审计文档；2026-08-29 用户已明确授权审查后归档到文档仓 `main`。
- 不要用 reset/checkout 清理文档或源码工作树；提交前仍须逐文件审查归属，并确保提交范围
  只包含 `/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization`。

### torch_npu 源码仓

- 路径：`/home/z50063656/Pass/src/torch_npu`；
- 基线 commit：`83cc452480c3546fd5cccf853bfe3a360ce9dbfc`；
- 共享树有大量累积和并发改动。与当前 Inductor 工作直接相关的 tracked 文件至少包括：
  `torch_npu/_inductor/{__init__.py,config.py,decomposition.py,lowering.py}`、
  `triton_experimental/{config.py,fx_passes.py,lowering.py,codegen/triton.py}` 及多个 FX/DVM 文件；
  目标测试也有修改/新增。
- 不要把共享树整体提交。P020 至 P026 都有对应 detached worktree、独立 wheel 和 venv，优先从
  隔离边界复核单个提案。

独立 venv 已存在 `p014-installed` 至 `p026-installed`；它们用于历史候选验证，不等于当前 Pass
主环境。

## 4. 已完成的有效工作（按新需求重新分类）

下表区分“直接回答社区原生 pass 任务”与“对后续仍有用的后端支撑”。这些工作的原始
日志、失败样本和环境标签都必须保留；“有效”不等于“已满足 2026-08-29 新完成标准”。

### 4.1 直接可用于社区原生 pass 主线的证据

| 对象 | 已完成的有效证据 | 当前边界/还缺什么 |
|---|---|---|
| T-056 静态 inventory | 已把 251 条概念级记录分流为 203 条 inherited-upstream、32 条 default-loader inactive、8 条其他后端、4 条上游显式关闭等路由，并记录 69 个 config 的绑定时机 | 这是候选地图，不是 203 条动态结论。现在需补“原生实现/注册 -> 社区测例 -> 去重验收单元”映射 |
| 上游 `mm_plus_mm` / experimental same-K | Conda Pass 下的正例命中 `_mm_plus_mm`、负例不误触发，generated code 和 extern 数量已检查；8/8 dtype/shape/layout/dynamic 功能网格正确，6/8 个代表配置 p50 收益超过 10%，transposed/dynamic 分别为 6.4%/8.74% 的中性收益；same-K backward 四个输入梯度正确 | 这是当前最完整的 experimental 原生 pass 动态证据。但尚未把它与社区 `test_pattern_matcher.py` 的直接用例建立可追溯映射，且当前 installed wheel 身份与当时已变化，所以先标记为“有效历史完整证据，待当前基线复核” |
| 上游 pad-mm family | 已证明 `pad_mm.py::check_device()` 只放行 CUDA/XPU，experimental 还额外关闭 shape padding；测试侧绕过 device gate 后，mm/bmm/addmm 正例均生成 `pad -> GEMM -> slice` 并正确，aligned 负例不触发 | 三路 paired p50 分别回退 72.65%/65.31%/120.63%，task 从 1 增至 5/5/7，峰值内存也增加。因此“保持 gate、不手写独立 padding Triton”是有效的失败/否决结论；仍需映射社区 `test_pad_mm.py` 用例后再按新矩阵关闭 |
| 上游 add+mm -> addmm / experimental | T-058/P-018 已证明默认关闭的两个上游 entry 可由现有 NPU addmm lowering 承接；独立 wheel 下 11/11 capability、正负例、dynamic/backward 和 live opt-out 通过，shape-A/unaligned host p50 改善 17.10%/13.52%，unaligned Event device p50/p99 改善 19.87%/19.10%，峰值内存不增 | 这是高价值原生 pass 适配候选，且无需手写 Triton addmm。但有效最终证据来自旧 Benchmark/独立 venv，未安装到当前 Pass 主环境；必须先对应社区 post-grad 测例并在当前 wheel 基线复核，不能直接写成当前产品已开启 |

**完成度口径**：按 `/home/z50063656/Pass/需求变更.md` 第 12 节的新标准，当前还没有一条
原生 pass 同时完成“社区测例映射 + 当前 Pass 主环境路由/数值/梯度 + pass-on/off 性能 +
证据归档”全部条件，因此 **新口径正式闭环数暂为 0**。这不表示旧工作无效：
`mm_plus_mm` 已有最接近完整的动态证据，pad family 已有充分的否决证据，experimental addmm
已有高置信隔离 wheel 候选；它们是 P0 映射和首批复核的直接输入，不应重做已有调研。

相关证据：

- `report/t056_triton_experimental_inventory_20260826.md`
- `report/p0_gate_first_run_20260820.md`
- `report/p0_sweep_function_matrix_20260820.md`
- `report/p0_sweep_performance_20260820.md`
- `report/p0_semantic_matrix_20260821.md`
- `report/t025_t026_pad_family_20260821.md`
- `report/t058_experimental_addmm_gate_20260826.md`

### 4.2 有效的基础设施和后端支撑（不直接计入原生 pass 完成数）

1. **Pass 主环境和可复现构建流程**：已建立 PyTorch/torch_npu/Triton Ascend/CANN/NPU
   版本指纹，torch_npu 源码 wheel 构建和 `--no-deps` 安装流程。wheel 身份冲突已显式登记，
   避免错误覆盖基线。
2. **触发与路由证据链**：`run_p0_gate_probe.py` 能以 fresh worker 分离 backend/case，并保存
   counter、pass 前后 FX、IR、`output_code.py`、extern/kernel 数量、fallback 和 graph-break 证据。
3. **单 pass A/B 方法**：已实现 fresh-process current/disabled、禁缓存、交错三轮、warmup/runs、
   p50/p99、首编、峰值内存和 NPU 占用复核。这套方法可直接用于社区测例迁移后的性能验收。
4. **backend 状态隔离结论**：T-057 证明 addmm check、config/guard、matmul fold 和 decomposition
   会在同进程回切 default 后残留；因此所有 experimental pass-on/off 必须用 fresh process。P-014 只
   修复 erfc 重入，没有关闭全部串态。
5. **T-055–T-072 后端稳定性工作**：P-014/P-016/P-017/P-019–P-026 以及 T-059–T-072 已形成
   有效的 correctness、lowering、codegen、autotune、wrapper 和 performance 证据。它们提高了
   `triton_experimental` 验收基础的可靠性，但大多不是社区原生 FX pass，不能按它们的
   feature family 数量计入主任务完成率。

### 4.3 有效的失败、否决和中性尝试

- pad-mm 绕 gate 后功能正确但性能/任务数/内存明显回退，所以保持 gate；这是有效否决，不是没完成。
- 多个旧 B2 pass 曾出现“数值误差为 0，但 alias/storage/stride 合同被破坏”；已形成不能只看数值的
  验收经验，可复用到原生 pass 迁移。
- `mm_plus_mm` transposed/dynamic 功能可用但收益不过 10% 门槛；这是 layout/dynamic 粒度的
  `supported-neutral` 证据，不能被 contiguous 收益覆盖。
- fresh Triton launcher 曾因 PyTorch C++20、Triton C++17 和 torch_npu/CANN header 不匹配失败。审计 shim
  只能用于归因，不能冒充正式产品环境已修复。

### 4.4 不能直接计入新主线完成度的历史工作

- default backend 的 P1 B2/B3、DVM/MLIR、B4 attention 和 P-013 保留为历史闭环，不直接计入
  `triton_experimental` 社区原生 pass 成功率。
- provenance、MSPTI/Event、wrapper allocator、range-tree/header tiling 等结果是有效专项证据，但不是
  “一条社区原生 pass 已验收”。
- 旧 Benchmark 和 P014–P026 独立 venv 结果不丢弃，但在当前 Pass 主环境复核之前不升级为当前
  installed verdict。

详细索引见 `current_status_and_background.md`、`outcome_index.md`、`change_control.md` 和
`report/t055_triton_experimental_enable_20260826.md` 至 `report/t072_matmul_fold_20260828.md`。

### 4.5 T-074 第一版静态索引（当前主线停点）

T-074 已完成第一版“上游候选 -> 社区测例 -> 去重验收单元”静态索引：

- T-056 的 203 条主候选全部进入索引，并额外纳入 pad mm/bmm/addmm 与 addmm 共 4 条
  `inherited-upstream-explicitly-disabled` 控制项，候选表总计 207 行；
- 203 条主候选只有 197 个唯一旧 `record_id`。T-074 保留旧字段，并为 207 个源码行生成
  唯一 `candidate_id`，没有用覆盖或合并掩盖重复记录；
- 207 行聚合为 188 个 provisional 验收单元，其中 158 个暂列 `yes-provisional`、30 个注册
  容器/扩展 hook 暂不进入分母；158 仍需人工审阅，**不是冻结分母**；
- 行级覆盖初分为 direct 120、indirect 33、`no-test-found` 54；单元级为 111/29/48；
- `mm_plus_mm`、pad mm/bmm/addmm 和 addmm 共 5 个首批单元已人工映射到社区测试，其余
  183 个单元仍标为 `generated-needs-human-review`；
- 生成器不导入 `torch`；`py_compile`、AST、计数/唯一性/文件存在性断言和连续两次确定性
  哈希检查均通过。本阶段没有运行 NPU、`torch.compile` 或性能测试，所有动态字段仍为
  `not-run-current-Pass`，所以新口径正式闭环数仍为 0。

直接继续工作的入口是：

- `report/t074_upstream_pass_test_index_20260829.md`
- `report/upstream_pass_test_index_20260829/candidate_test_index.csv`
- `report/upstream_pass_test_index_20260829/acceptance_units.csv`
- 审计生成器：
  `/home/z50063656/Pass/inductor_pass_npu_audit/t074_build_upstream_test_index.py`

## 5. T073 当前停点（已降为次级支线）

目标是 `TE-DEC-002` softmax backward no-FMA decomposition。尚无正式 T073 报告，也没有
专属关闭条目（E-198 已用于 T-074），但运行文件和原始结果已经存在：

- runner：`/home/z50063656/Pass/inductor_pass_npu_audit/t073_softmax_no_fma_*.py` 与
  `run_t073_softmax_no_fma_*.sh`；
- 结果：`/home/z50063656/Pass/inductor_pass_npu_audit/results/t073_softmax_no_fma_20260828/`；
- static contract：18/18；synthetic contract：11/11、7 个 case；
- NPU on/off 功能结果除一次设备子进程启动超时外均通过；失败的
  `rank4_bf16_off/result.json` 已由 `rank4_bf16_off_v2` 重试通过，原失败应保留为环境中性证据；
- on 路径 generated code 不再出现 FMA，代表性能图由 2 个 Triton kernel 降为 1 个；
- 三轮旧环境预结果的 device P50 中位约为 `0.50090 -> 0.23345 ms`，约改善 `53.39%`；
  device P99 中位约为 `0.53406 -> 0.25574 ms`，约改善 `52.11%`；peak allocated 从
  `119,548,416 B` 降为 `60,822,016 B`。这些数字来自 P026 独立 venv，并且 PyTorch import
  仍指向 Benchmark source，只能作为高潜力候选，不能作为 Pass 最终 verdict。

T073 是有价值的 experimental decomposition/codegen 高潜力支线，原始证据必须保留；但它尚未
对应到当前“社区原生 pass -> 社区测例”的完成单元，因此不再是紧接着的主线。恢复 T073 时
仍然不能直接写“优化成功”：应先迁移 runner 环境入口、在 Pass 复核 decomposition 选择、
correctness、generated code 和至少一轮性能哨兵，再决定是否完整重跑三轮并形成正式报告。

## 6. 建议的新对话执行顺序

1. 阅读 `/home/z50063656/AGENTS.md`、`/home/z50063656/Pass/需求变更.md`、本文件、
   `change_control.md` 和 `current_status_and_background.md`。
2. T-074 已完成第一版静态生成，不要重建另一套平行索引。先审阅
   `candidate_test_index.csv` 中 54 条 `no-test-found` 和 33 条 indirect 行，修正一对多
   注册、生成式 pattern 和测试归属，再冻结真正的 eligible 分母。
3. 审阅时保留 `candidate_id` 与旧 `record_id` 两层键；不要把 203 行压成 197 个旧 ID，
   也不要把 158 个 provisional 单元提前写成完成分母。
4. 首批 5 个单元的社区映射已经回填。以 `mm_plus_mm` 为第一个当前 Pass 动态迁移对象，
   pad family 作为历史性能否决控制，addmm 作为历史隔离 wheel 收益候选；不要重做已有静态调研。
5. 从同时具有正例、负例和具体 counter 的社区用例中选择后续批次，做最小 NPU +
   `options={"npu_backend": "triton_experimental"}` 迁移。
6. 启动动态测试前，从 `/home/z50063656/tmp` source `Pass/activate_pass.sh`，只读核验 Python/import
   路径、installed wheel identity 和 NPU 空闲卡。未登记变更不得安装当前 `dist/` 同名 wheel。
7. 每个动态用例先证明目标 pattern 命中、数值/梯度、generated code 和无非预期 fallback；通过后
   再做 fresh-process pass-on/pass-off paired 性能，不先全量重跑 T059–T073 feature family。
8. T073 和 P020–P026 的 Pass 主环境复核保留为次级交付支线；只在 P0 索引或主线直接需要这些
   后端能力时提前恢复。
9. 本批文档按 E-199 的用户授权归档到文档仓。后续产品 diff 仍按提案拆分审查；没有新的
   明确授权，不提交或推送任何产品源码仓。

## 7. 可直接粘贴给新对话的提示词

```text
请先阅读：
1. /home/z50063656/AGENTS.md
2. /home/z50063656/Pass/需求变更.md
3. /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/HANDOFF_20260828_PASS_ENV.md
4. /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/change_control.md

当前主环境必须是 /home/z50063656/Pass/activate_pass.sh 激活的 Conda Pass；所有测试从
/home/z50063656/tmp 发起。不要使用 Benchmark/env.sh，不要直接重装 dist 中的同名 torch_npu
wheel，不要清理共享源码树。T-074 第一版已经把 T-056 的 203 条上游候选和 4 条显式关闭
控制项映射为 188 个 provisional 去重单元；158 个 eligible 候选仍不是冻结分母。先审阅
54 条 no-test-found 与 33 条 indirect 行，再以已经人工回填的 `mm_plus_mm` 为首个社区用例做
NPU `triton_experimental` 最小迁移；pad family 和 experimental addmm 保留为首批控制/候选。
旧 Benchmark/独立 venv 结果保留为历史证据，不自动升级为当前 Pass 主环境 verdict。
```
