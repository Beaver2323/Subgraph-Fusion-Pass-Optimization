# T-075 首批 Acceptance Unit 静态映射复核

> 更新时间：2026-08-31 18:34 CST（UTC+08:00）
> 状态：`completed-first-wave-static-mapping-await-gpu-reference`
> 运行边界：本任务只做源码、community test 与数据结构复核；未运行 GPU、NPU、
> `torch.compile` 或性能测试。

## 1. 结论

T-074 v1 中首批 5 个 provisional 单元经人工复核后仍收敛为 5 个 upstream optimization
contracts，没有机械地按 registration 数量拆分：

| Acceptance unit | Contract 决策 | Variant 数 | Community test 数 | 当前状态 |
| --- | --- | ---: | ---: | --- |
| `AU-post-grad-mm-plus-mm` | same-K、different-K、输出形状不匹配属于同一 contract 的正负 variants | 3 | 1 | 等待 GPU/reference |
| `AU-pad-mm-mm` | 独立的二维 mm padding/slice/stride contract | 4 | 4 | 等待 GPU/reference |
| `AU-pad-mm-bmm` | 独立的 batched mm padding/slice contract | 4 | 3 | 等待 GPU/reference |
| `AU-pad-mm-addmm` | 独立的 addmm padding/bias/slice contract | 5 | 3 | 等待 GPU/reference |
| `AU-post-grad-addmm` | 两种 add 与 mm 操作数顺序共享一个 addmm replacement contract | 4 | 2 | 等待 GPU/reference |

本轮共登记 20 个 variants、13 个 community test 引用。所有单元均为
`mapped-static-await-gpu-reference`，分母状态均为 `pending-reference`；正式冻结分母和正式闭环数
仍为 0。T-074 的 207 行 candidate inventory、188 个 provisional units 和 158 个 provisional
eligible 统计保持原样，没有覆盖或重生成 v1 CSV。

## 2. 基线与事实源

| 项目 | 冻结值或状态 |
| --- | --- |
| PyTorch | `release/2.14`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `master`，commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| T-074 candidate CSV | `report/upstream_pass_test_index_20260829/candidate_test_index.csv` |
| candidate CSV SHA256 | `7c79d4c586d34e8af5d77ea548f66307a7f64e62437e3659b98693a88f271da8` |
| 主要事实源 | PyTorch community tests；后续由 GPU/reference baseline 固化运行预期 |
| 辅助证据 | registration、源码 guard、历史 NPU runner/report |

torch_npu 源码树存在任务开始前已有的 experimental overlay，本轮没有修改或提交该源码树。
特别是 addmm 在冻结 commit 基线下由 `disable_addmm_fusion=True` 关闭，而当前工作树 overlay
表现为默认启用并提供 live opt-out；当前 Pass installed wheel 身份冲突尚未重验。因此本轮只记录
source candidate/control 差异，不给出 installed NPU product verdict。

## 3. Contract 与 Variant 决策

### 3.1 `mm_plus_mm`

- `same-K-positive` 与 `different-k-positive-pattern-fallback` 都满足 post-grad pattern 的核心
  M/N 输出合同，因此不拆成两个 acceptance units；
- different-K 可以命中 pattern，但 `tuned_mm_plus_mm()` 的完整 size equality guard 会把它回退为
  two-mm-plus-add。这里必须分别记录“pattern 是否命中”和“runtime path 是否融合”；
- 输出 shape 不匹配是同一 contract 的 negative variant；community test 的全局 counter 仍可能被
  其他 add/mm pattern 增加，GPU runner 必须补目标 pattern counter 或 generated-code 证据，不能仅按
  suite 总计数判断目标命中。

### 3.2 pad mm/bmm/addmm

- 三者拥有独立 search pattern、replacement 和运行语义，继续保留为三个 acceptance units；
- `gen_register_replacement` 为 training 与 inference 展开 registration，但这是一个 contract 的
  注册承载 variants，不是两个 acceptance units；
- NPU `triton_experimental` 产品基线由 `disable_pad_mm=True` 显式关闭。正常 product run 的
  `matched=0` 应归类为 expected disabled/guarded，不能写成“不支持”；只有单独标记的 diagnostic
  gate-bypass 才能讨论候选能力。

### 3.3 add+mm → addmm

- `add(mm(...), bias)` 与 `add(bias, mm(...))` 两个对称 registration 共用一个 handler 和
  replacement contract，因此保留一个 acceptance unit；
- matrix bias、vector bias、两种 operand order 是 positive variants；不可广播/批量 bias 与
  Python 或 symbolic scalar 是 negative variants；
- community `test_addmm` 中两个顺序预期产生两个 matches，但它们仍属于同一 contract，计数不能
  被误读为两个 acceptance units。

## 4. T-074 v1 到 T-075 的可审计修正

1. 单元总数不变，首批仍为 5 个；变化发生在 contract/variant 和证据角色，不是数字重算。
2. `test_no_autocast_in_pad_bmm_joint_graph_pass` 从 direct trigger 证据降级为
   `related-regression`：它验证 dtype/gradient 行为，不能单独证明目标 padding pattern 命中。
3. pad-mm 映射新增输出 stride 回归；pad-bmm 新增 static fp16 `test_pad_batch`；pad-addmm
   新增 2D bias broadcast 正例。
4. CUDA-specific 的 beta=0 mismatched-bias 负例只保留为 reference 证据，不能未经运行直接投射为
   NPU 行为。
5. addmm 明确区分冻结 commit control、dirty source overlay 和 installed wheel 三个状态，防止把
   源码可见性当成当前安装态结论。

## 5. 落盘产物

- `upstream/manifest.schema.json`：acceptance-unit 结构合同；
- `upstream/manifest.yaml`：首批 5 个已人工审核单元及 20 个 variants；
- `upstream/pass_map.yaml`：5 条 T-074 candidate 到 registration/test/unit 的多对多映射；
- `scripts/validate_tracker_data.py`：不导入 `torch` 的标准库静态校验器。

YAML 文件采用 JSON-compatible YAML，GPU 机器无需额外安装 PyYAML。静态校验同时确认 manifest、
mapping、T-074 candidate ID、community test 文件和测试方法存在，并输出：

```text
tracker_data_validation=OK
acceptance_units=5
mapped_candidates=5
variants=20
community_tests=13
denominator_frozen=0
torch_imported=0
```

## 6. 下一步

下一任务是 T-076：只针对这 5 个已静态审核单元生成 manifest-driven GPU/reference runner 和
人工操作说明。执行顺序固定为：

1. 在 GPU 上优先直接运行 community 原生测例/原生 helper，确认是否真实命中；
2. 只有 direct 方式受设备常量、backend 注入或 artifacts 采集阻塞时，才使用不改变图与预期行为的
   最小 adapter；
3. GPU baseline 有效后再进入当前 Pass 环境的 NPU comparison；
4. 48 个 `no-test-found` 与 29 个 indirect 单元继续做静态人工审核，但不得在 reference 缺失前
   冻结 denominator。
