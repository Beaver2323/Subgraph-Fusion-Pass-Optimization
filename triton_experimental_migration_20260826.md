# triton_experimental 需求变更与迁移基线（2026-08-26）

## 1. 最终执行口径

用户明确要求：运行环境保持原样，只复用
`inductor-meta-worktree` 的工作流程；本任务后续负责的后端为
`torch_npu/_inductor/triton_experimental`。

因此本项目不会安装或切换到 meta 仓库描述的 PyTorch 2.13/CANN 9.1 环境，也不会执行其
`env.sh`、`init.md` 或 `quick-init.md`。实际运行合同保持：

| 项目 | 保持值 |
|---|---|
| 工作目录 | `/home/z50063656/Pass` |
| 环境入口 | `/home/z50063656/Benchmark/env.sh` |
| Python | 3.11.15，Conda `benchmark-py311` |
| PyTorch | `2.14.0a0+git8e86e0a`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0a0+git83cc452`，commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| torch_npu 安装 | 当前源码构建 wheel，继续 `--force-reinstall --no-deps` |
| Triton runtime | `triton.__version__=3.2.0`；现有 metadata 不一致风险继续保留 |
| CANN / NPU | CANN 9.0.1；8 x Ascend 910B2 |
| 测试 cwd | `/home/z50063656/tmp`；不得在 torch_npu 源码树内 import torch |

meta 工作流只在 `/tmp/inductor-meta-worktree-readonly` 做了只读浅克隆，HEAD 为
`6d3bd619f910cf00380e4cd1d90b9d63a68de488`；旧外挂后端只在
`/tmp/npu_inductor_2_13_readonly` 做了只读浅克隆，HEAD 为
`0661e649a9a8673ca438181cb978a1b978b222df`。二者都不是安装位置，不参与运行时导入。

## 2. 复用的工作流程

从 meta 工作流吸收以下流程，不吸收它的版本和目录：

1. 每个 shell 先 source 项目唯一环境入口；本项目对应 `Benchmark/env.sh`。
2. 修改前固定源码 commit、安装 wheel、设备、cache、候选范围、回滚边界和验收门槛。
3. 先验证 backend 入口和隔离，再做 pass inventory、correctness、codegen、paired performance。
4. Python 快速迭代可以用于诊断，但正式结论必须来自当前源码构建、`--no-deps` 安装的 wheel。
5. 根据修改位置决定同步 Python、重建 torch_npu wheel或重编 Triton；不因编译错误修改无关源码。
6. 正式验证使用 fresh cache/fresh process，并同时检查数值、generated code、P50/P99、首次编译、task 和峰值内存。
7. 所有环境失败、无效隔离和中性尝试原样记录，不计入产品 pass 成功率。

由于全局项目规则要求测试从 `/home/z50063656/tmp` 发起，meta 工作流中
`cd pytorch/test/_inductor` 的命令模板不会原样照搬。实际做法是 source 现有环境后，从 tmp
以绝对路径启动测试。

## 3. 当前源码中的 experimental 后端

当前 torch_npu commit 已包含在树后端，共 15 个 Python 文件，关键入口为：

- `torch_npu/_inductor/__init__.py:_load_triton_experimental_backend()`：恢复上游
  Inductor baseline，注册 experimental decompositions，再调用 `_activate()`；
- `triton_experimental/__init__.py:_activate()`：注册独立 device/codegen、fallback、autotune；
- `triton_experimental/overrides.py:apply_npu_overrides()`：安装 experimental 配置、FX 和
  codegen patch；
- `triton_experimental/fx_passes.py`：当前直接拥有 int→float→int elide、Max(1) loop-merge，
  并默认关闭 pad-mm 和 add+mm→addmm；
- `triton_experimental/codegen/`、`npu_triton_heuristics.py`：独立 Triton codegen、tiling、
  autotune 和 launcher。

选择入口仍使用当前源码支持的三种方式：per-compile
`options={"npu_backend": "triton_experimental"}`、
`torch._inductor.config.npu_backend` 或在 import torch_npu 前设置
`TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。本项目优先使用 per-compile，避免污染同一
进程的 default 对照。

## 4. 既有成果的迁移分类

此前 T-011 至 T-054 都在当前版本环境中完成，但主要验证 default backend。证据和方法保留，
verdict 不直接升级为 experimental 结论。

| 既有对象 | experimental 源码事实 | 迁移决定 |
|---|---|---|
| P-013 pattern 5 guard | 只在 `_load_triton_backend()` 调用；experimental loader 不调用 | guard 不自动生效；先重新测原 rewrite/codegen/性能，再决定是否在 experimental 增加同类 gate |
| different-K `mm_plus_mm` | pattern extra-check 明确要求 backend 为 `default`；experimental 不注册候选 | standalone kernel/验收脚本可复用；pattern、lowering与性能必须重新接入和验证 |
| B2 27 个 NPU custom pass | default loader调用 `pre_grad_custom_pass_fuc()`/`post_grad_custom_pass_fuc()`；experimental `_activate()` 不调用 | 旧 27 条为 default 历史结论；逐条判断是否应迁移，不批量打开 |
| add+mm→addmm | experimental 默认 `disable_addmm_fusion=True` 并修改已注册 entry 的 check | default 的 beneficial 数据只提供优先级；先验证关闭原因和 experimental lowering/codegen，不直接解除 gate |
| pad mm/bmm/addmm | experimental 默认 `disable_pad_mm=True` | 与旧 65–121% 回退方向一致，先保持关闭，只做最小隔离哨兵 |
| DVM/MLIR | 属于其他 backend loader | 保留历史结果，不纳入 experimental 成功率 |
| 文档、探针、fresh-process paired 方法 | 与 backend 无关 | 直接复用方法和证据格式 |

旧 251 行矩阵继续作为 current-version/default-backend 历史基线，不覆盖、不清零。experimental
需要增加独立 backend 维度或新矩阵：统计对象应包括“在 experimental 激活后实际运行的上游
Inductor pass、experimental 自有 FX pass、codegen/loop rewrite 与显式关闭 gate”。

## 5. 恢复后的执行顺序

原暂停检查点中预告的 default pattern-21 T-055 在执行前被本次需求变更取代，不再直接启动。
新的顺序为：

1. T-055：当前安装 wheel 的 experimental 三入口、default 隔离、generated wrapper marker、
   float32/int64 pointwise correctness 冒烟。三入口 12/12 通过，但 experimental 回切 default
   因 erfc decomposition 重复注册失败；P-014 单行 cleanup 修复已完成 source overlay 双向切换
   1/1，installed wheel 复验待共享 `triton.py` diff 可安全隔离后执行。
2. T-056：已按当前 2.14 源码建立 251 行 experimental route overlay、69 项 config 引用表和
   35 个 feature family；静态结果见 `report/t056_triton_experimental_inventory_20260826.md`。
3. T-057：backend 全局状态串态、int-float-int 与 GELU approximate 已完成。前者登记 P-015
   设计阻断；int-float-int 由 Float32 ON/OFF 锁定数值错误和三 dtype alias 错误，P-016 已 source
   默认关闭；GELU installed-none 合同失败，P-017 current source 已通过 FP32/FP16/BF16、非法参数
   和 generated-code 验证恢复合同。P-016/P-017 都是 wheel pending。T-058 addmm 首个 FP16
   vector-bias cohort 已证明单 extern addmm 相对 mm+Triton add 的 p50/p99 中位数改善
   22.17%/14.05%，峰值内存相同；后续 11/11 capability 与 unaligned 第二性能 cohort 通过，
   P-018 current source 已默认启用并保留可恢复 opt-out，source gate 通过、wheel/host-tail pending。
   下一步做 permute-gather、outer rsplit；pad-mm 只保留 disabled sentinel。
4. 只有出现明确的不可用、错误 codegen 或性能回退，才登记 P-014 及后续产品修改；修改优先
   落在 `triton_experimental/`，不得把 default loader 的 patch 直接复制过去。
5. 任何性能结论继续执行同机、空闲卡、fresh-process paired，并保留原图 fallback。

## 6. 当前停止边界

迁移初始化本身没有安装环境或删除此前 P-013 工作。后续 P-014/P-016/P-017/P-018 已按
document-first 流程形成四个最小 source 修改并通过 source 验证，但没有构建或安装 wheel；当前 installed
P-013 wheel、P-012 回滚 wheel 和源码快照继续由 `PAUSED_CHECKPOINT_20260826_P013.md` 管理。
共享 tree 其他未安装 diff 可隔离前，禁止把它们随 P-014/P-016/P-017/P-018 一并打包。
