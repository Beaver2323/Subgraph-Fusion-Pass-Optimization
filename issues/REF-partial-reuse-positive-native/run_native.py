#!/usr/bin/env python3
"""T-078 partial reduction reuse 正例的原生入口预检与适配调度。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


WORK = Path("/home/z50063656/tmp")
SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py"
)
METHODS = [
    "TestPatternMatcher.test_successful_partial_reuse_case0",
    "TestPatternMatcher.test_successful_partial_reuse_case1",
    "TestPatternMatcher.test_successful_partial_reuse_case2",
]

parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument(
    "--adapter",
    action="store_true",
    help="仅在原生入口阻断后运行同一社区方法的最小 NPU 适配",
)
mode.add_argument(
    "--source-fix",
    action="store_true",
    help="从 torch_npu 工作树加载 min2 正式修复并运行同一社区方法",
)
mode.add_argument(
    "--neighbor",
    choices=("nan-propagation",),
    help="在正式源码下运行 min reduction 的语义近邻验证",
)
args = parser.parse_args()
if Path.cwd().resolve() != WORK:
    raise SystemExit("必须从 /home/z50063656/tmp 启动")

root = WORK / "t078-npu-results" / "REF-partial-reuse-positive-native"
root.mkdir(parents=True, exist_ok=True)
prefix = (
    f"neighbor-{args.neighbor}-"
    if args.neighbor
    else "source-fix-"
    if args.source_fix
    else ("adapter-" if args.adapter else "direct-")
)
run = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
env = dict(
    os.environ,
    TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
    TORCH_DEVICE_BACKEND_AUTOLOAD="1",
    TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
    TORCHINDUCTOR_COMPILE_THREADS="1",
    TORCH_COMPILE_DEBUG="1",
    TORCH_COMPILE_DEBUG_DIR=str(run / "debug"),
    TORCH_TRACE=str(run / "trace"),
    TORCHINDUCTOR_CACHE_DIR=str(run / "inductor-cache"),
    TRITON_CACHE_DIR=str(run / "triton-cache"),
)
if args.source_fix or args.neighbor:
    overlay = Path(__file__).parents[2] / "runners" / "t078_source_overlay"
    previous_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{overlay}{os.pathsep}{previous_pythonpath}"
        if previous_pythonpath
        else str(overlay)
    )
code = """import json, os, runpy, sys
from pathlib import Path
import torch, torch_npu
from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
register_inductor_npu()
metadata = {'backend': _InductorNpuRegistry._loaded_backend, 'torch': torch.__version__,
            'torch_npu': torch_npu.__version__, 'device_available': torch.npu.is_available(),
            'torch_commit': torch.version.git_version, 'cwd': os.getcwd(),
            'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}
Path(sys.argv[2]).write_text(json.dumps(metadata, indent=2) + '\\n')
assert metadata['backend'] == 'triton_experimental'
sys.argv = [sys.argv[1], *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
command = [
    sys.executable,
    "-c",
    code,
    str(SOURCE),
    str(run / "environment.json"),
    *METHODS,
]
if args.adapter or args.source_fix:
    command = [
        sys.executable,
        str(Path(__file__).with_name("npu_adapter.py")),
        "--artifact-dir",
        str(run),
    ]
    if args.source_fix:
        command.append("--source-fix")
elif args.neighbor:
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_neighbors.py")),
        "--artifact-dir",
        str(run),
        "--case",
        args.neighbor,
    ]

with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
    try:
        result = subprocess.run(
            command,
            cwd=WORK,
            env=env,
            stdout=out,
            stderr=err,
            timeout=900,
        )
        status = result.returncode
    except subprocess.TimeoutExpired:
        status = 124

summary = {
    "source_test": (
        f"tracker-derived::{args.neighbor}"
        if args.neighbor
        else f"{SOURCE}::TestPatternMatcher.test_successful_partial_reuse"
    ),
    "direct_args": METHODS,
    "return_code": status,
    "tracking_mode": (
        "source-fix-neighbor-verification"
        if args.neighbor
        else "source-fix-verification"
        if args.source_fix
        else ("adapter" if args.adapter else "direct")
    ),
    "raw_logs": str(run),
    "body_or_assertions_modified": False,
    "source_kind": "tracker-derived-neighbor" if args.neighbor else "community",
    "reference_raw_evidence_pending": True,
}
(run / "direct_result.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False), flush=True)
raise SystemExit(status)
