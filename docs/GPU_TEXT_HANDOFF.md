# GPU 原文 handoff 导出、复制与恢复指南

> 更新时间：2026-09-06 06:43 CST（UTC+08:00）
> 适用任务：T-076～T-080 及后续复用统一 GPU reference runner 的任务
> 目标：在 GPU 服务器不能直接推 Git、不能传二进制时，用一个可复制 JSON 回传可校验的 FX、日志、生成代码和 IR 原文

## 1. 默认行为

从本版本开始，一键入口：

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" --task T-078 --gpu 2
```

会在功能 reference 完成后自动生成 **1.1 原文 handoff**：

```text
/data/z50063656/tmp/t078-reference-results/latest-text-handoff.json
```

不需要查找 `reference-<timestamp>`。`latest-text-handoff.json` 始终指向最后发布的一轮；控制台同时
打印真实 `run_dir=`，用于审计并发运行。

两种格式的边界如下：

| 格式 | 产生方式 | 内容 | 能否恢复 FX/日志正文 |
| --- | --- | --- | --- |
| `1.0` | 手工调用导出器且不加 `--include-raw-text` | 环境、summary、逐 case 状态、文件大小和 SHA256 | 不能 |
| `1.1` | GPU 一键入口默认；或手工加 `--include-raw-text` | 1.0 全部字段，加已登记 UTF-8 原文和二进制缺项清单 | 能恢复文本原文 |

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

如果完整 artifacts 仍在 GPU 服务器，可直接把旧 run 重新导出为 1.1，不会重新运行 GPU：

```bash
export RUN_DIR="$(readlink -f /data/z50063656/tmp/t078-reference-results/latest)"
export RAW_HANDOFF=/data/z50063656/tmp/t078-reference-handoff-v1.1.json

cd /data/z50063656/tmp
python "${TRACKER_ROOT}/scripts/export_reference_text.py" \
  --run-dir "${RUN_DIR}" \
  --include-raw-text \
  --output "${RAW_HANDOFF}"

python "${TRACKER_ROOT}/scripts/import_reference_text.py" \
  --input "${RAW_HANDOFF}" \
  --validate-only
```

导出器使用“只新建、不覆盖”策略。`RAW_HANDOFF` 已存在时应换一个文件名或先人工保留旧文件；它
不会覆盖原始 run、已有 handoff、软链接或任何原证据。

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

如果网页编辑器因文件过大拒绝保存，不能截断、拆改或删除日志后伪装成完整 handoff；应保留
`wc -c`、错误信息和原 artifacts，再决定使用受控文件传输或另行设计分片格式。

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
既有文件。随后按实际 case 查看：

```bash
export RESTORED_RUN=/home/z50063656/tmp/gpu-reference-imports/text-import-.../reference-...

sed -n '1,240p' "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native/fx_before.txt"
sed -n '1,240p' "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native/fx_after.txt"
find "${RESTORED_RUN}/cases/REF-addcdiv-fma-bitwise-native" \
  -type f \( -name 'output_code.py' -o -name '*.ttir' -o -name '*.ttgir' -o -name '*.ptx' \) \
  -print
```

恢复只是数据读取。导入器不会 `import`、`exec`、运行或编译回传的 `.py`、PTX、Triton IR；回执
`import_receipt.json` 固定记录 `code_executed=false`。恢复成功也不自动把 case 判为 PASS，仍需按
reference summary、测试数、skip、correctness、FX 和源码 revision 完成验收。

## 5. 包含内容与缺项

1.1 会嵌入 `artifact_inventory.json` 登记且后缀为下列类型的严格 UTF-8 原文：

```text
json/jsonl, txt/log, py, csv, yaml/yml, ptx, ttir/ttgir, ll/mlir,
dot, c/cpp/cu, h/hpp, s/asm, html, md
```

因此 `fx_before.txt`、`fx_after.txt`、`stdout.log`、`stderr.log`、`reference_result.json`、
`output_code.py` 和常见 Triton/LLVM/PTX 文本可以离线恢复。非 UTF-8、含 NUL 或非文本后缀的文件
不会塞进 JSON；它们保留在 `raw_text_transfer.omitted_files`，包括原路径、字节数、SHA256 和原因。

`all_registered_artifacts_embedded=false` 通常只说明存在 cubin、so 等二进制，并不表示已嵌入文本
不完整。这个 JSON 不是二进制 artifacts 的完整归档，也不能替代需要二进制反汇编或重新执行的
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
