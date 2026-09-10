#!/usr/bin/env python3
"""在占用GPU前拒绝没有case的已审核批次。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cases = plan.get("cases")
    if not isinstance(cases, list):
        raise ValueError("reference plan cases必须是列表")
    if not cases:
        print(
            f"gpu_task_runnable=0 task={plan.get('task_id')} "
            "reason=reviewed-no-gpu-ready-units"
        )
        return 4
    print(f"gpu_task_runnable=1 task={plan.get('task_id')} cases={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
