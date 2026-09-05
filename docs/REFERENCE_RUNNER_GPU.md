# T-076 GPU/reference Runner 人工操作说明

> 更新时间：2026-09-06 05:20 CST（UTC+08:00）
> 状态：GPU 环境与文本 handoff 已复核；13/13 direct cases 均 passed 且 `reference_valid=true`。
> 核心规则：先取得 direct 结果；没有 direct blocker 证据，不创建或运行 adapter。

## 1. GPU 机器环境合同

当前 GPU 机器环境固定为：

| 项目 | 合同 |
| --- | --- |
| 执行用户 | 登录账号 `z00824525`，具有 sudo 权限；`HOME=/data/z50063656` |
| GPU | 8 × NVIDIA A100-SXM4-80GB，SM 8.0 |
| 宿主驱动 | `550.54.15`，保持不变 |
| CUDA Toolkit | 12.6.3，安装到 `/data/z50063656/cuda-12.6` |
| CUDA compat | 当前未启用；Driver API 12.4 + CUDA 12.6 minor compatibility 已通过 kernel 验真 |
| cuDNN | pip `nvidia-cudnn-cu12` 9.25.1 |
| Python | `/data/z50063656/envs/PassGPURef` pip venv，Python 3.12 |
| Triton | 3.8.0 |
| PyTorch source | `/data/z50063656/src/pytorch`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| 测试工作目录 | `/data/z50063656/tmp`，必须位于 PyTorch 源码树外 |
| 结果目录 | `/data/z50063656/tmp/t076-reference-results` |

根分区只剩少量空间，CUDA、venv、源码、缓存、构建产物和测试结果都必须写入 `/data`。不要
使用 `apt install cuda-*` 把 Toolkit 写入根分区，不升级宿主驱动，不修改
`/usr/local/cuda -> /usr/local/cuda-12.4`。NPU 的 `/home/z50063656/Pass/activate_pass.sh`
不能代替 GPU 环境。

runner 还要求：

- PyTorch source working tree 为 clean；
- 当前 Python 的 `torch.version.git_version` 与冻结 source commit 完全一致；
- CUDA GPU、cuDNN 和 Triton 可用；
- A100 执行前选择空闲卡，不终止或迁移其他用户任务。

## 2. 环境安装与激活

CUDA 12.6.3 使用 NVIDIA runfile，只安装 Toolkit、不安装驱动：

```bash
mkdir -p /data/z50063656/{downloads,cuda-12.6,envs,src,tmp,pip-cache,cache,triton-cache,inductor-cache,torch-extensions}

export TMPDIR=/data/z50063656/tmp
cd /data/z50063656/downloads
wget -c https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit \
  --toolkitpath=/data/z50063656/cuda-12.6
```

创建独立 pip venv 并准备冻结源码。若目标目录已存在，先核对其来源，不要直接覆盖：

```bash
export TMPDIR=/data/z50063656/tmp
export PIP_CACHE_DIR=/data/z50063656/pip-cache

python3.12 -m venv /data/z50063656/envs/PassGPURef
source /data/z50063656/envs/PassGPURef/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install nvidia-cudnn-cu12 cmake ninja mkl-static mkl-include

git clone --recursive --branch release/2.14 \
  https://github.com/pytorch/pytorch.git /data/z50063656/src/pytorch
git -C /data/z50063656/src/pytorch switch --detach \
  8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b
git -C /data/z50063656/src/pytorch submodule sync --recursive
git -C /data/z50063656/src/pytorch submodule update --init --recursive

cd /data/z50063656/src/pytorch
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make triton
```

上述下载、venv、pip cache 和源码都位于 `/data`；不要执行 `apt autoremove` 清理共享机器。

每次登录后执行：

```bash
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
export CUDNN_LIBRARY="${CUDNN_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_COMPAT_DIR CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH

export PYTHON="$(command -v python)"
export PASS_TRACKER_WORK_DIR=/data/z50063656/tmp
```

冻结 PyTorch 必须从 source 构建；构建命令固定为：

```bash
cd /data/z50063656/src/pytorch
export USE_CUDA=1 USE_CUDNN=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}"
python -m pip install -e . -v --no-build-isolation
```

不要在 PyTorch 源码目录中执行导入 `torch` 的探针或测试。

## 3. 更新仓库并核对 commit

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

git -C /data/z50063656/src/pytorch rev-parse HEAD
git -C /data/z50063656/src/pytorch status --short
```

HEAD 必须为 `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`，`status --short` 必须无输出。

日常复跑推荐使用统一入口；它会自动激活环境、校验、运行、导出文本并维护 `latest`：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-076 --gpu 2
```

默认共享；加 `--wait-gpu` 每 1 秒检查启动条件，独占需加 `--exclusive`。
完整说明见[通用一键说明](GPU_TASK_RUNNER.md)。下文底层命令不含等卡/协作锁，仅保留用于排障。

## 4. 先做零设备静态校验

```bash
export PASS_TRACKER_WORK_DIR=/data/z50063656/tmp
export PYTHON=/data/z50063656/envs/PassGPURef/bin/python
cd /data/z50063656/tmp

bash "${TRACKER_ROOT}/scripts/run_reference_all.sh" \
  --pytorch-root /data/z50063656/src/pytorch \
  --validate-only
```

预期关键输出：

```text
reference_plan_validation=OK acceptance_units=5 cases=13 community_tests=13 variants=20 executed_variants=14 non_executed_variants=6
torch_imported=0 gpu_executed=0
```

## 5. 执行全部原生社区测例

先用 `nvidia-smi` 查看所选物理卡的空余显存（功能测试可以共享），再设置当前 shell 及子进程的设备号：

```bash
export CUDA_VISIBLE_DEVICES=2  # 换成所选物理 GPU 编号
cd /data/z50063656/tmp

bash "${TRACKER_ROOT}/scripts/run_reference_all.sh" \
  --pytorch-root /data/z50063656/src/pytorch \
  --output-root /data/z50063656/tmp/t076-reference-results
```

执行合同：13 个 case 直接调用冻结 commit 中的社区测试；每个 case 使用 fresh Python process、
独立 cache/debug/trace；失败、skip 或超时后继续其余 case；首批不运行 benchmark。runner 会在
`environment.json` 中保存执行用户、CUDA/compat/Conda 路径、`cuDriverGetVersion()`、宿主驱动、
`nvcc`、PyTorch/cuDNN/Triton 和 GPU 指纹。环境或 commit 不一致时会先保存证据再停止。

## 6. 单 case 重跑

```bash
cd /data/z50063656/tmp
bash "${TRACKER_ROOT}/scripts/run_reference_all.sh" \
  --pytorch-root /data/z50063656/src/pytorch \
  --output-root /data/z50063656/tmp/t076-reference-results \
  --case REF-mm-plus-mm-native
```

不要修改社区图、shape、dtype、正负例语义或 expected match 来让失败消失。

## 7. 结果、打包与回传

每轮生成独立的 `reference-<timestamp>` 目录，包含环境、manifest/plan snapshot、summary，以及
每个 case 的 metadata、result、FX、日志、debug 和 structured trace。`reference_valid=true`
同时要求原生测试通过并捕获计划要求的 FX before/after；skip、未发现测试或缺失 FX 仍为 invalid。

```bash
cd /data/z50063656/tmp/t076-reference-results
export RUN_DIR="$(readlink -f latest)"
tar -czf reference-latest.tar.gz -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
sha256sum reference-latest.tar.gz > reference-latest.tar.gz.sha256
```

退出码非零也要回传完整 run 目录，不先删除失败 case 或大日志。runner 不自动提交 artifacts。

若机器禁止 Git/二进制文件上传，统一入口已经自动调用通用文本导出器。直接使用稳定入口：

```bash
export RESULT_ROOT=/data/z50063656/tmp/t076-reference-results
export RUN_DIR="$(readlink -f "${RESULT_ROOT}/latest")"
export TEXT_HANDOFF="$(readlink -f "${RESULT_ROOT}/latest-text-handoff.json")"

cd /data/z50063656/tmp
python -m json.tool "${TEXT_HANDOFF}" >/dev/null
sha256sum "${TEXT_HANDOFF}"
wc -c "${TEXT_HANDOFF}"
```

该文件只包含完整环境、suite 摘要、逐 case 审核字段、关键证据文件的大小/SHA256，以及文本
payload 的稳定 SHA256；不包含大体积 debug 正文，不修改原始 run。通过终端复制该 JSON 即可
完成受限机器的结构化交接。原始 run 仍须保留在 GPU 机器，供后续按哈希追溯。

## 8. Direct blocker 与 adapter 决策

| Direct 结果 | 后续动作 |
| --- | --- |
| `passed` 且 `reference_valid=true` | 保留 direct，不创建 adapter |
| big-GPU/device harness skip | 先确认硬件合同，再评估只移除 harness 的薄 adapter |
| backend/device 常量阻塞 | 保留原 traceback，允许只注入 device/backend 的 adapter |
| 仅 artifact capture 缺失 | 保留原图和原断言，只补采集薄层 |
| 正确性、counter 或 FileCheck 断言失败 | 按 upstream/environment failure 分析，不改预期 |
| 图、shape、dtype、正负例需改变才能通过 | 禁止称为最小 adapter，退回 mapping 审核 |

任何 adapter/extracted case 进入 `reference_plan.yaml` 前，都必须记录对应 direct blocker artifact。
