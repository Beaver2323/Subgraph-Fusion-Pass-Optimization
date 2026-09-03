#!/usr/bin/env python3
"""聚合 T-077 B2B GEMM 功能门禁与社区网格 capability 探针。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    functional = {}
    for mode in ("baseline", "candidate"):
        path = args.run_dir / "functional" / mode / "capability_result.json"
        if path.exists():
            functional[mode] = json.loads(path.read_text(encoding="utf-8"))

    grid = []
    for path in sorted((args.run_dir / "community-grid").glob("*/capability_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        observation = result["observations"][0]
        stderr = path.with_name("stderr.log")
        autotune = None
        if stderr.exists():
            matches = re.findall(
                r'\{"num_choices".*\}', stderr.read_text(errors="replace")
            )
            if matches:
                autotune = json.loads(matches[-1])
        grid.append(
            {
                "point": path.parent.name,
                **observation,
                "autotune": autotune,
            }
        )

    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-077",
        "acceptance_unit_id": "AU-b2b-gemm",
        "backend": "triton_experimental",
        "status": "capability-assessed",
        "functional": {
            mode: {
                "tests_run": data["tests_run"],
                "successful": data["successful"],
                "matched": sum(
                    item["target_count"] > 0 for item in data["observations"]
                ),
                "template_selected": sum(
                    item["template_selected"] for item in data["observations"]
                ),
            }
            for mode, data in functional.items()
        },
        "community_grid": {
            "executed_points": len(grid),
            "matcher_accepted": sum(item["target_count"] > 0 for item in grid),
            "heuristic_rejected": sum(item["target_count"] == 0 for item in grid),
            "template_selected": sum(item["template_selected"] for item in grid),
            "points": grid,
        },
    }
    output = args.run_dir / "capability_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"capability_summary={output}")
    print(
        "b2b_capability="
        f"points:{len(grid)} "
        f"matched:{result['community_grid']['matcher_accepted']} "
        f"selected:{result['community_grid']['template_selected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
