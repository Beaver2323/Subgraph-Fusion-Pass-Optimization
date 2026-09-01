# T-077 GPU/reference Runner 操作说明

> 更新时间：2026-09-02 05:45 CST（UTC+08:00）
> 状态：第二波 5 个 acceptance units、11 个 direct cases、17 个 variants 已完成静态映射与零设备校验，等待 GPU 执行。
> 执行原则：先运行冻结 PyTorch commit 中的原生社区测例；direct 无效时只回传证据，不在 GPU 机器临时改图或写 adapter。

## 1. 本轮范围

| Acceptance unit | Direct cases | Variants | 主要合同 |
| --- | ---: | ---: | --- |
| `AU-apply-gumbel-max-trick` | 1 | 1 | 等价 Gumbel-max 采样与统计分布 |
| `AU-b2b-gemm` | 6 | 6 | 左/右结合、pointwise 正例及 pattern/shape 负例 |
| `AU-decompose-mem-bound-mm-decompose-bmm` | 1 | 3 | 大 batch 小矩阵 BMM 的阈值正负例 |
| `AU-decompose-mem-bound-mm-decompose-mm` | 2 | 6 | fp32/bf16 autocast 下 MM 的阈值正负例 |
| `AU-decompose-mem-bound-mm-decompose-addmm` | 1 | 1 | 动态大 M、小 K/N addmm 分解 |

合计 5 个待冻结单元、11 个原生 case、17 个全部由动态测例覆盖的 variant。CPU-only
`test_decompose_bmm_cpu` 和 `test_decompose_mm_cpu` 不属于本轮 GPU reference 合同。

## 2. 前置环境

沿用 T-076 已验真的 GPU 环境，不重新安装 CUDA、驱动、cuDNN、PyTorch 或 Triton：

| 项目 | 固定值 |
| --- | --- |
| GPU | 8 × NVIDIA A100-SXM4-80GB，SM 8.0 |
| 驱动 | `550.54.15`，不升级 |
| Toolkit | `/data/z50063656/cuda-12.6`，CUDA 12.6.3 |
| Python | `/data/z50063656/envs/PassGPURef/bin/python`，Python 3.12 |
| PyTorch source | `/data/z50063656/src/pytorch` |
| PyTorch commit | `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| 工作目录 | `/data/z50063656/tmp` |
| 输出根目录 | `/data/z50063656/tmp/t077-reference-results` |

每次登录后执行：

```bash
export HOME=/data/z50063656
export TMPDIR=/data/z50063656/tmp
export PIP_CACHE_DIR=/data/z50063656/pip-cache
export XDG_CACHE_HOME=/data/z50063656/cache
export TRITON_CACHE_DIR=/data/z50063656/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/data/z50063656/inductor-cache
export TORCH_EXTENSIONS_DIR=/data/z50063656/torch-extensions

source /data/z50063656/envs/PassGPURef/bin/activate
export CUDA_HOME=/data/z50063656/cuda-12.6
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDNN_ROOT="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn"
export CUDNN_INCLUDE_DIR="${CUDNN_ROOT}/include"
export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_COMPAT_DIR CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH

export PYTHON="$(command -v python)"
export PASS_TRACKER_WORK_DIR=/data/z50063656/tmp
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
export PYTORCH_ROOT=/data/z50063656/src/pytorch
export RESULT_ROOT=/data/z50063656/tmp/t077-reference-results
```

不要在 PyTorch 或 torch_npu 源码树内执行 `import torch` 或测试。

## 3. 更新并静态校验

```bash
git -C "${TRACKER_ROOT}" pull --ff-only origin main
git -C "${PYTORCH_ROOT}" rev-parse HEAD
git -C "${PYTORCH_ROOT}" status --short

cd /data/z50063656/tmp
bash "${TRACKER_ROOT}/scripts/run_t077_reference_all.sh" \
  --pytorch-root "${PYTORCH_ROOT}" \
  --validate-only
```

PyTorch HEAD 必须等于冻结 commit，`status --short` 必须无输出。静态校验预期：

```text
reference_plan_validation=OK acceptance_units=5 cases=11 community_tests=11 variants=17 executed_variants=17 non_executed_variants=0
torch_imported=0 gpu_executed=0
```

## 4. 执行完整 GPU reference

先用 `nvidia-smi` 选择无其他用户进程的空闲物理卡，然后只限制当前 shell：

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=<空闲GPU编号>

cd /data/z50063656/tmp
bash "${TRACKER_ROOT}/scripts/run_t077_reference_all.sh" \
  --pytorch-root "${PYTORCH_ROOT}" \
  --output-root "${RESULT_ROOT}"
```

runner 为每个 case 启动 fresh Python process，并建立独立 cache、debug、structured trace；单个
失败、skip 或 timeout 不会中断其余 case。`REF-decompose-addmm-dynamic-native` 保留上游
`M=19494144`，单独允许 7200 秒，不得为了缩短运行时间改小输入。

只重跑一个 case：

```bash
cd /data/z50063656/tmp
bash "${TRACKER_ROOT}/scripts/run_t077_reference_all.sh" \
  --pytorch-root "${PYTORCH_ROOT}" \
  --output-root "${RESULT_ROOT}" \
  --case REF-decompose-addmm-dynamic-native
```

完整 suite 只有在 11/11 case 均 `passed` 且 `reference_valid=true` 时才可冻结 T-077 denominator。

## 5. 文本回传

GPU 机器禁止 Git 或二进制上传时，继续使用通用文本导出器：

```bash
export RUN_DIR="${RESULT_ROOT}/reference-<timestamp>"
export TEXT_HANDOFF="${RESULT_ROOT}/reference-<timestamp>-text-handoff.json"

cd /data/z50063656/tmp
python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${RUN_DIR}" \
  --output "${TEXT_HANDOFF}"

python -m json.tool "${TEXT_HANDOFF}" >/dev/null
sha256sum "${TEXT_HANDOFF}"
wc -c "${TEXT_HANDOFF}"
```

请复制完整 JSON 文本。失败时也保留并导出整轮目录；不要删除失败 case，不要把 skip 记作 PASS。

## 6. Direct blocker 决策

- `passed + reference_valid=true`：保持 direct，不创建 adapter；
- 测试失败、skip、no-tests、缺失 FX：原样回传日志与 artifacts；
- 只有明确属于 device 常量、backend 注入或 artifact 采集阻塞时，NPU 控制节点才设计最小 adapter；
- 不得更改 graph、shape、dtype、dynamic、forward/backward、统计容差或正负例来制造通过。
