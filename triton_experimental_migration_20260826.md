# triton_experimental 需求变更与迁移基线（2026-08-26）

> **2026-08-28 追加口径**：用户已将主工作虚拟环境纠正为 Conda `Pass`。下表和后续新测试
> 流程按 Pass 执行；文中关于 Benchmark 的既有实验叙述仅表示当时真实环境，不得据此继续
> 启动新任务。

> **2026-08-29 主线校准**：下文 T-055 至 T-073 的执行顺序是历史 experimental feature-family
> 记录，不再代表当前优先级。当前主线为 T-074 社区原生 pass/pattern 测例索引与 NPU 迁移；
> 第一版结果见 `report/t074_upstream_pass_test_index_20260829.md`。

## 1. 最终执行口径

用户明确要求：运行环境保持原样，只复用
`inductor-meta-worktree` 的工作流程；本任务后续负责的后端为
`torch_npu/_inductor/triton_experimental`。

因此本项目不会安装或切换到 meta 仓库描述的 PyTorch 2.13/CANN 9.1 环境，也不会执行其
`env.sh`、`init.md` 或 `quick-init.md`。实际运行合同保持：

| 项目 | 保持值 |
|---|---|
| 工作目录 | `/home/z50063656/Pass` |
| 环境入口 | `/home/z50063656/Pass/activate_pass.sh` |
| Python | 3.11.15，Conda `Pass` (`/home/z50063656/envs/Pass`) |
| PyTorch | `2.14.0a0+git8e86e0a`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0a0+git83cc452`，commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| torch_npu 安装 | installed 记录哈希 `263ffec2...2792704`；dist 同名文件已是 `3909fd64...c2d4989`，登记前禁止重装 |
| Triton runtime | `triton.__version__=3.2.0`；现有 metadata 不一致风险继续保留 |
| CANN / NPU | CANN 9.0.1；8 x Ascend 910B2 |
| 测试 cwd | `/home/z50063656/tmp`；不得在 torch_npu 源码树内 import torch |

meta 工作流只在 `/tmp/inductor-meta-worktree-readonly` 做了只读浅克隆，HEAD 为
`6d3bd619f910cf00380e4cd1d90b9d63a68de488`；旧外挂后端只在
`/tmp/npu_inductor_2_13_readonly` 做了只读浅克隆，HEAD 为
`0661e649a9a8673ca438181cb978a1b978b222df`。二者都不是安装位置，不参与运行时导入。

## 2. 复用的工作流程

从 meta 工作流吸收以下流程，不吸收它的版本和目录：

1. 每个 shell 先 source 项目唯一环境入口；本项目当前对应 `Pass/activate_pass.sh`。
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
   和 generated-code 验证恢复合同。P-016 已完成独立 wheel 的默认 OFF/late opt-in，P-017 也已
   完成六组 GELU/非法参数/P-014 近邻的独立 wheel 验证。T-058 addmm 首个 FP16
   vector-bias cohort 已证明单 extern addmm 相对 mm+Triton add 的 p50/p99 中位数改善
   22.17%/14.05%，峰值内存相同；后续 11/11 capability 与 unaligned 第二性能 cohort 通过，
   P-018 current source 已默认启用并保留 match-time live opt-out。独立 wheel 的
   default/late-opt-out/restore、11/11 capability 和两组 paired 性能均通过；shape-A/unaligned
   host p50 改善 17.10%/13.52%，Event device p50/p99 改善 19.87%/19.10%，host-tail 保留监控。
   T-059 permute-gather 随后以无源码修改归档；P-019 outer rsplit 独立 wheel 的 target UT 5/5、
   NPU 矩阵 9/9 和三轮 paired 通过，device P50/P99 改善中位数 `32.67%/28.58%`；P-020
   int64 dtype-aware fallback 独立 wheel 的 target UT 6/6、NPU 矩阵 8/8 和正确基线性能通过。
   T-062 随后验证 generate-list fallback 合同和 12/12 功能矩阵；全局 `ceil` generate 因
   standalone 回退否决，mixed 单 kernel 收益只登记为上下文感知机会。T-063 又完成
   TE-CG-005 六项门控的静态契约、NPU 14/14 配置、18/18 数值对照和三轮性能矩阵；默认值
   保留，device p99/host tail 继续监控。T-064 再以 P-021/P-022 修复 zero-X promotion 和
   flattened R-tree 错算；安装态 UT 6/6、7/7、NPU 18/18 通过，flatten 因 device P50 回退
   147.96% 保持默认关闭。T-065 随后关闭 TE-CG-008：五项配置 10/10 正确性，三个有效开关
   18/18 paired 性能通过；input-stride/odometer 默认保留，align-8 中性保留，unify/padding
   两项无效配置只登记清理。T-066 随后关闭 group/all-block dispatch：group config 无消费者，
   但 48-core 实际 group codegen 保留；P-023 对齐 auto-blockify 与 backend env gate，安装态
   target UT 3/3、近邻 2/2、candidate 两态和 dynamic smoke 通过。T-067 再关闭 autotune
   配置族：默认公式路径保留，P-024 删除 `autotune_enhance` 导入快照并完成安装态 L0 53/53、
   NPU 开/关两态验证。T-068 随后保留快 `55.73%` 的 NPU direct allocation，并以 P-025
   让 WorkspaceArg 局部过滤后委托上游 memory planner；安装态 contract、11/11 allocation UT、
   pointwise/rsplit workspace 均通过。上述任务均不写重复 Triton kernel；pad-mm 只保留
   disabled sentinel。T-069 又验证自动/预加载 MSPTI 精确采集、三轮 Event paired 和运行态
   失败回退；MSPTI autotune wall 改善 `86.84%`，peak 少 `255,984,640 B`，默认优先路径与
   Event fallback 均保留。T-070 随后证明 Max(1,size) fold 将动态卷积 pointwise 从 3 轴合为
   1 轴，并承接空输出和关闭态多轴 rank-mismatch；性能中性、峰值不变，不新增产品修改。
   T-071 随后确认 recursive dict tag 当前上游默认已为 False；深层 NPU 训练两态正确，三轮
   CPU guard 量化关闭 fast path 的 P50 开销中位 `36.57%`，默认安全覆盖保留。T-072 随后
   以 P-026 将广泛 matmul fold 收窄到精确 seq-first stride；目标 P50 中位改善 `5.53%`、
   peak 下降 `83.09%`，三类扩大作用域回退恢复上游。按当时计划下一项为
   T-073/TE-DEC-002。
4. 只有出现明确的不可用、错误 codegen 或性能回退，才登记 P-014 及后续产品修改；修改优先
   落在 `triton_experimental/`，不得把 default loader 的 patch 直接复制过去。
5. 任何性能结论继续执行同机、空闲卡、fresh-process paired，并保留原图 fallback。

## 6. 当前停止边界

迁移初始化本身没有安装环境或删除此前 P-013 工作。后续 P-014/P-016/P-017/P-018/P-019/P-020
已按 document-first 流程形成六个隔离修改并完成各自 detached worktree、专属 wheel 与独立 venv
闭环；T-062/T-063/T-065 未产生产品修改，T-064 新增 P-021/P-022 两个隔离修复并完成
wheel/venv 闭环，T-066 新增 P-023、T-067 新增 P-024、T-068 新增 P-025 并分别完成独立
wheel/venv 闭环；T-069/T-070/T-071 均未修改产品源码并已完成 P-025 安装态验证；T-072
新增 P-026 并完成正式 wheel/clean venv 闭环；按当时计划下一项为 T-073/TE-DEC-002。共享 installed
P-013 wheel、P-012
回滚 wheel 和源码快照继续由 `PAUSED_CHECKPOINT_20260826_P013.md` 管理。共享 tree 其他未安装
diff 仍不得混入任一专属 wheel。
