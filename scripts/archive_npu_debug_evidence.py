#!/usr/bin/env python3
"""将 NPU 修复前/后 torch_compile_debug 文本证据归档到对应 issue。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


EVIDENCE_NAMES = (
    "fx_graph_readable.py",
    "fx_graph_transformed.py",
    "ir_pre_fusion.txt",
    "ir_post_fusion.txt",
    "output_code.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_models(source_root: Path) -> list[Path]:
    models = sorted(
        path
        for path in source_root.glob(
            "debug/torch_compile_debug/run_*/torchinductor/model*"
        )
        if path.is_dir()
    )
    if not models:
        raise SystemExit(f"未找到 torch_compile_debug model 目录：{source_root}")
    return models


def archive_stage(issue: Path, stage: str, source_root: Path) -> dict:
    destination = issue / "evidence" / stage
    if destination.exists():
        raise SystemExit(f"目标已存在，拒绝覆盖：{destination}")
    destination.mkdir(parents=True)
    inventory = []
    for model in collect_models(source_root):
        # 单次 source root 理论上只有一个 run；仍保留 run 名避免未来多 run 撞名。
        run_name = next(
            parent.name for parent in model.parents if parent.name.startswith("run_")
        )
        model_destination = destination / run_name / model.name
        copied = 0
        for name in EVIDENCE_NAMES:
            source = model / name
            if not source.is_file():
                continue
            model_destination.mkdir(parents=True, exist_ok=True)
            target = model_destination / name
            shutil.copy2(source, target)
            inventory.append(
                {
                    "path": str(target.relative_to(issue)),
                    "source": str(source),
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )
            copied += 1
        if copied == 0:
            raise SystemExit(f"model 目录没有目标证据文件：{model}")
    return {
        "stage": stage,
        "source_root": str(source_root),
        "models": len({Path(item["path"]).parent for item in inventory}),
        "files": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-label", required=True)
    parser.add_argument("--after-label", required=True)
    parser.add_argument("--meaning", required=True)
    args = parser.parse_args()

    issue = args.issue.resolve()
    if not issue.is_dir() or issue.parent.name != "issues":
        raise SystemExit(f"不是仓库 issues 下的目录：{issue}")
    before = args.before.resolve()
    after = args.after.resolve()
    for source in (before, after):
        if not source.is_dir():
            raise SystemExit(f"来源目录不存在：{source}")

    stages = [
        archive_stage(issue, "修复前", before),
        archive_stage(issue, "修复后", after),
    ]
    generated_at = datetime.now().astimezone().isoformat()
    manifest = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "issue": issue.name,
        "evidence_kind": "npu-torch-compile-debug-before-after",
        "stages": stages,
    }
    manifest_path = issue / "evidence/evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# NPU 修复前后编译证据

> 归档时间：{generated_at}

## 对照定义

- 修复前：{args.before_label}
- 修复后：{args.after_label}
- 证据含义：{args.meaning}

## 文件怎么读

每个 `run_*/model*` 目录按 Inductor 编译阶段保存五类原始文本：

1. `fx_graph_readable.py`：进入后端处理的可读 FX 图；
2. `fx_graph_transformed.py`：Inductor 图变换后的 FX 图；
3. `ir_pre_fusion.txt`：scheduler fusion 前的 Inductor IR；
4. `ir_post_fusion.txt`：scheduler fusion 后的 Inductor IR；
5. `output_code.py`：最终生成并实际执行的 wrapper/kernel 代码。

不是每个模型都会生成全部五类文件；缺失表示该次编译没有产出该阶段文本，不能补造。
本目录仅归档文本证据，不包含 `.so`、二进制 kernel、cache 或 trace。每个文件的原始绝对路径、
字节数和 SHA256 见 `evidence_manifest.json`。

## 使用边界

“修复前/修复后”只按上面的对照定义解释。产品门禁类修复中，修复前可能是显式重开风险路径，
修复后是默认关闭路径；这类证据证明最终产品行为，而不是两个不同源码 commit 的二分结果。
"""
    (issue / "evidence/README.md").write_text(readme, encoding="utf-8")
    print(
        f"archive=OK issue={issue.name} "
        f"before_files={len(stages[0]['files'])} after_files={len(stages[1]['files'])}"
    )


if __name__ == "__main__":
    main()
