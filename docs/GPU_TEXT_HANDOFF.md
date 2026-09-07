# GPU 原文 handoff 导出、复制与恢复指南

> 更新时间：2026-09-07 08:25 CST（UTC+08:00）
> 适用任务：T-076～T-080 及后续复用统一 GPU reference runner 的任务
> 目标：在 GPU 服务器不能直接推 Git、不能传二进制时，默认用短评审包回传可校验的摘要与 FX；完整文本归档按需导出

## 1. 默认行为

从本版本开始，一键入口：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2
```

会在功能 reference 完成后自动生成 **1.3 压缩评审 handoff**：

```text
/data/z50063656/tmp/t078-reference-results/latest-text-handoff.json
```

统一入口默认使用单行 JSON 以减少文本体积，并以 96 KiB 为网页单文件经验阈值。未超限时打印
`handoff_upload_mode=single-file`；超限时自动生成分片并打印 `handoff_upload_mode=split` 与唯一的
`handoff_upload_input=.../manifest.json`。这里的阈值只控制上传形式，不改变 handoff 内容和哈希。

不需要查找 `reference-<timestamp>`。`latest-text-handoff.json` 始终指向最后发布的一轮；控制台同时
打印真实 `run_dir=`，用于审计并发运行。

三种 profile 与历史格式的边界如下：

| 格式/profile | 产生方式 | 内容 | 用途 |
| --- | --- | --- | --- |
| `1.0 summary` | `--profile summary` | 环境、summary、逐 case 状态、文件大小和 SHA256 | 只看结论，不能查看 FX 正文 |
| `1.3 review` | 一键入口默认；`--profile review` | 1.0 加 FX 前后、case 元数据、结果、benchmark、inventory；失败 case 加日志 | GPU/NPU 功能与性能评审，推荐网页回传 |
| `1.2 archive` | `--profile archive` | 嵌入全部已登记 UTF-8 日志、生成代码和 IR | 深度排障/审计，通常较大 |
| `1.1 archive` | 旧参数 `--include-raw-text` | 1.2 的未压缩兼容格式 | 仅兼容旧流程 |

旧的 1.0 紧凑包仍可用于摘要复核，但不能据此查看 GPU 实际 FX 图、生成代码或完整日志；导入器会
明确拒绝把 1.0 冒充可恢复证据。

## 2. GPU 侧执行与校验

以 T-078、物理 GPU 2 为例：

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
git -C "${TRACKER_ROOT}" pull --ff-only origin main

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2 \
  --wait-gpu
```

完成后只用固定入口：

```bash
export RESULT_ROOT=/data/z50063656/tmp/t078-reference-results
export TEXT_HANDOFF="$(readlink -f "${RESULT_ROOT}/latest-text-handoff.json")"

cd /data/z50063656/tmp
python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${TEXT_HANDOFF}" \
  --validate-only
wc -c "${TEXT_HANDOFF}"
sha256sum "${TEXT_HANDOFF}"
```

成功校验会打印：

```text
handoff_validation=OK run_id=reference-... restorable_text_files=... code_executed=false
```

这里的 `sha256sum` 是传输文件本身的哈希；JSON 内的 `payload_sha256` 是与空白排版无关的整包内容
哈希。接收端以导入器同时复核整包哈希、逐文件大小/SHA256 和 inventory 绑定为准。

### 2.1 已有历史 run 不重跑 GPU

如果完整 artifacts 仍在 GPU 服务器，可直接把旧 run 重新导出为 1.3 review，不会重新运行 GPU：

```bash
export RUN_DIR="$(readlink -f /data/z50063656/tmp/t078-reference-results/latest)"
export REVIEW_HANDOFF=/data/z50063656/tmp/t078-reference-handoff-review-v1.3.json

cd /data/z50063656/tmp
python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${RUN_DIR}" \
  --profile review \
  --compact \
  --output "${REVIEW_HANDOFF}"

python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${REVIEW_HANDOFF}" \
  --validate-only
```

T-076/T-077 已完成正式结论，但需补齐 1.3 review 学习/审计正文。旧 runner 没有 `latest` 软链接时，
直接使用已知 run 目录；以下命令只导出，不重跑 GPU，并生成单行 JSON：

```bash
export T076_RUN=/data/z50063656/tmp/t076-reference-results/reference-20260901T180826+0800
export T077_RUN=/data/z50063656/tmp/t077-reference-results/reference-20260902T125636+0800

python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${T076_RUN}" \
  --profile review \
  --compact \
  --output /data/z50063656/tmp/t076-handoff-review-v1.3.json

python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${T077_RUN}" \
  --profile review \
  --compact \
  --output /data/z50063656/tmp/t077-handoff-review-v1.3.json
```

分别校验后，完整复制到 `results/incoming/T-076/text-handoff.json` 和
`results/incoming/T-077/text-handoff.json`。T-076 当前仓库文件在第 1001 行截断，必须整体替换；
T-077 当前没有 handoff 文件。若上述 run 已被清理，才需要重新执行对应 GPU 任务。

review 包保留 FX 对照所需正文，并为未传输的成功日志、生成代码、IR 和二进制保留 inventory
大小与 SHA256；完整文件仍留在 GPU 原 run。导出器使用“只新建、不覆盖”策略。输出已存在时应换一个文件名或先人工保留旧文件；它
不会覆盖原始 run、已有 handoff、软链接或任何原证据。

只有深度排障需要全部文本时才改用：

```bash
python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${RUN_DIR}" \
  --profile archive \
  --compact \
  --output /data/z50063656/tmp/t078-reference-handoff-archive-v1.2.json
```

### 2.2 超限自动分片

一键入口会自动比较单行 handoff 与 96 KiB 阈值；超过时把同一 handoff 拆为一个 manifest 和多个
小 JSON，每个分片内部再将 Base64 按短行保存，无需先尝试上传大文件。运行结束只看：

```text
handoff_upload_mode=split
handoff_upload_input=/data/.../latest/text-handoff-parts/manifest.json
```

已有旧 run 需要独立补生成时可显式使用同一机制：

```bash
export PARTS_DIR=/data/z50063656/tmp/t079-handoff-review-parts-v1.3

python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${RUN_DIR}" \
  --profile review \
  --compact \
  --split-output-dir "${PARTS_DIR}"

python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${PARTS_DIR}/manifest.json" \
  --validate-only

find "${PARTS_DIR}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

默认每片承载 48 KiB 原始 JSON；经 Base64 和 JSON 包装后通常约 66 KiB。这个默认值来自
T-079 的实际回传：约 260 KiB 的首版分片仍被当前 GitHub 网页通道拒绝，因此进一步缩小。
将 `manifest.json` 和全部 `part-*.json` 原样创建到例如：

```text
results/incoming/T-079/text-handoff-parts/
├── manifest.json
├── part-0001.json
├── part-0002.json
└── ...
```

控制节点直接把 manifest 交给导入器；导入器会验证分片顺序、归属、偏移、逐片哈希、重建整包
SHA256 和 handoff 内部全部证据，无需人工合并：

```bash
python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${TRACKER_ROOT}/results/incoming/T-079/text-handoff-parts/manifest.json" \
  --validate-only
```

一键入口仍保留完整单文件作为本地证据；只有超过阈值才创建
`latest/text-handoff-parts/manifest.json`。已有旧 run 可用上述命令补生成分片，无需重跑 GPU。

## 3. 通过文本复制上传 GitHub

GPU 侧先打印原文件，复制从第一个 `{` 到最后一个 `}` 的全部内容：

```bash
cat "${TEXT_HANDOFF}"
```

不要把 shell 提示符、`sha256sum`、runner 日志或 Markdown 围栏混入 JSON。然后在 GitHub 网页中
创建或更新对应任务文件，例如：

```text
results/incoming/T-078/text-handoff.json
```

建议提交说明写明任务号和 JSON 内的 `reference_summary.run_id`。同一路径的新 Git 提交会保留旧版
历史；如需并存多轮，也可另建带 run ID 的 JSON 文件。实际 handoff 在本地默认被 `.gitignore`
忽略，但通过 GitHub 网页提交后会成为受版本控制文件，控制节点 `git pull` 可以取得。

如果网页编辑器因单文件过大拒绝保存，不能截断、删日志或手工改 JSON；应使用 2.2 节的受控
分片格式。默认约 66 KiB 仍不适用时，可在导出命令增加 `--split-part-bytes 16384`，将单片进一步
降到约 23 KiB；导入和校验方式不变。

## 4. 控制节点校验、恢复与查看 FX

拉取网页提交后，从 `/home/z50063656/tmp` 操作，避免在 torch_npu 源码树内导入运行环境：

```bash
export TRACKER_ROOT=/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
export INPUT="${TRACKER_ROOT}/results/incoming/T-078/text-handoff.json"
export RESTORE_ROOT=/home/z50063656/tmp/gpu-reference-imports

git -C "${TRACKER_ROOT}" pull --ff-only origin main
cd /home/z50063656/tmp

python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${INPUT}" \
  --validate-only

python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${INPUT}" \
  --output-root "${RESTORE_ROOT}"
```

第二条命令打印唯一的 `restored_run=`。恢复目录强制位于 tracker 仓库外，每次创建新目录，不覆盖
既有文件。review 包按实际 case 查看 FX：

```bash
export RESTORED_RUN=/home/z50063656/tmp/gpu-reference-imports/text-import-.../reference-...

sed -n '1,240p' "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native/fx_before.txt"
sed -n '1,240p' "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native/fx_after.txt"
```

仅当导入的是 archive 包时，才会恢复可查询的生成代码与 IR：

```bash
find "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native" \
  -type f \( -name 'output_code.py' -o -name '*.ttir' -o -name '*.ttgir' -o -name '*.ptx' \) \
  -print
```

review 包默认不包含 `output_code.py`、PTX 或 Triton IR 正文；这些文件只保留 inventory 哈希，
需要查看时从 GPU 原 run 取证或另导出 archive。恢复只是数据读取。导入器不会 `import`、`exec`、运行或编译任何回传正文；回执
`import_receipt.json` 固定记录 `code_executed=false`。恢复成功也不自动把 case 判为 PASS，仍需按
reference summary、测试数、skip、correctness、FX 和源码 revision 完成验收。

## 5. 包含内容与缺项

1.3 review 固定嵌入：

```text
environment.json, reference_summary.json；
每个 case 的 artifact_inventory.json、benchmark.json、fx_before.txt、
fx_after.txt、metadata.json、reference_result.json；
失败或无效 case 的 stdout.log、stderr.log。
```

通过 case 的日志、生成代码、IR、缓存和所有二进制只登记原路径、字节数、SHA256 与
`review-profile-hash-only` 原因。1.2 archive 才会继续嵌入 inventory 登记且后缀为下列类型的
严格 UTF-8 原文：

```text
json/jsonl, txt/log, py, csv, yaml/yml, ptx, ttir/ttgir, ll/mlir,
dot, c/cpp/cu, h/hpp, s/asm, html, md
```

archive 中的 `fx_before.txt`、`fx_after.txt`、`stdout.log`、`stderr.log`、`reference_result.json`、
`output_code.py` 和常见 Triton/LLVM/PTX 文本可以离线恢复。非 UTF-8、含 NUL 或非文本后缀的文件
不会塞进 JSON；它们保留在 `raw_text_transfer.omitted_files`，包括原路径、字节数、SHA256 和原因。

review 的 `all_registered_artifacts_embedded=false` 是正常状态，表示评审范围外文件只保留哈希；
archive 中该值为 false 通常说明存在 cubin、so 等二进制。handoff 不能替代需要二进制反汇编或重新执行的
场景。原文总量上限为 64 MiB；超限时导出失败而不是静默截断。

## 6. 完整性和安全边界

- 整包 `payload_sha256` 防止文本复制缺失或字段被改动；
- 每份恢复文件必须同时匹配登记的字节数和 SHA256；
- case inventory 与 handoff 清单交叉绑定，不能通过同时伪造正文自报哈希来替换原证据；
- 拒绝绝对路径、`..`、`.git`、反斜杠、控制字符、重复路径、文件/目录冲突和源软链接；
- 恢复目标不得位于 tracker 仓库内，且所有输出均只新建；
- 导入器仅解析 JSON 并写回文本，从不执行回传代码。

任何校验失败都应保留原文件和错误信息，重新复制或回到 GPU 原 artifacts 核查；不得手工修改
`payload_sha256`、逐文件 SHA256 或 inventory 来让校验“通过”。
