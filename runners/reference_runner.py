#!/usr/bin/env python3
"""Run manifest-mapped PyTorch community tests on a GPU reference host."""

from __future__ import annotations

import argparse
import ast
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 顶层必须是 object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_command(
    command: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_output(root: Path, *args: str) -> str:
    completed = run_command(["git", "-C", str(root), *args], cwd=root)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git 命令失败")
    return completed.stdout.strip()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def split_nodeid(nodeid: str) -> tuple[str, str]:
    try:
        relative, qualname = nodeid.split("::", 1)
    except ValueError as error:
        raise ValueError(f"community nodeid 格式错误: {nodeid}") from error
    return relative, qualname


def python_qualnames(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            result.add(".".join([*self.scope, node.name]))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            result.add(".".join([*self.scope, node.name]))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    Visitor().visit(tree)
    return result


def validate_contract(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    pytorch_root: Path,
) -> dict[str, int]:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("reference plan schema_version 不符合 runner 合同")
    if not re.fullmatch(r"T-[0-9]{3}", plan.get("task_id", "")):
        raise ValueError("reference plan task_id 必须形如 T-077")
    if manifest.get("schema_version") != plan.get("manifest", {}).get(
        "schema_version"
    ):
        raise ValueError("reference plan 与 manifest schema_version 不一致")
    expected_commit = manifest["source_baselines"]["pytorch"]["commit"]
    if plan["manifest"]["pytorch_commit"] != expected_commit:
        raise ValueError("reference plan 与 manifest 的 PyTorch commit 不一致")
    if plan["execution_policy"]["order"] != (
        "native-community-test-first-then-minimal-adapter-if-required"
    ):
        raise ValueError("执行顺序必须先原生社区测例，再考虑最小适配")
    if plan["execution_policy"]["failure_behavior"] != "continue-and-record":
        raise ValueError("单 case 失败必须继续并记录")

    units = {
        unit["acceptance_unit_id"]: unit
        for unit in manifest["acceptance_units"]
    }
    manifest_tests = {
        (unit_id, test["nodeid"])
        for unit_id, unit in units.items()
        for test in unit["community_tests"]
    }
    seen_case_ids: set[str] = set()
    seen_tests: set[tuple[str, str]] = set()
    covered_variants: set[tuple[str, str]] = set()
    qualname_cache: dict[Path, set[str]] = {}

    for case in plan["cases"]:
        case_id = case["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"重复 case_id: {case_id}")
        seen_case_ids.add(case_id)
        unit_id = case["acceptance_unit_id"]
        if unit_id not in units:
            raise ValueError(f"case 引用了未知 acceptance unit: {unit_id}")
        test_key = (unit_id, case["source_test"])
        if test_key not in manifest_tests:
            raise ValueError(f"case 不是 manifest community test: {case_id}")
        if test_key in seen_tests:
            raise ValueError(f"community test 被重复执行: {test_key}")
        seen_tests.add(test_key)

        mode = case["tracking_mode"]
        if mode not in {"direct", "adapter", "extracted"}:
            raise ValueError(f"{case_id} tracking_mode 非法: {mode}")
        if mode != units[unit_id]["tracking"]["reference_mode"]:
            raise ValueError(f"{case_id} 与 manifest reference_mode 不一致")
        if mode != "direct":
            if not case.get("entrypoint") or not case.get("direct_blocker"):
                raise ValueError(
                    f"{case_id} 进入 {mode} 前必须记录 entrypoint/direct_blocker"
                )
            blocker = case["direct_blocker"]
            required = {"direct_case_id", "artifact_ref", "classification"}
            if required - blocker.keys():
                raise ValueError(f"{case_id} direct_blocker 字段不完整")
        if mode != "direct" and case.get("direct_args"):
            raise ValueError(f"{case_id} 非 direct case 不得设置 direct_args")
        if set(case["required_artifacts"]) != {"fx-before", "fx-after"}:
            raise ValueError(f"{case_id} 必须采集 FX before/after")
        if not case["expected_assertions"]:
            raise ValueError(f"{case_id} expected_assertions 不能为空")
        timeout = case.get(
            "timeout_seconds", plan["execution_policy"]["default_timeout_seconds"]
        )
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"{case_id} timeout 必须为正整数")
        known_variants = {
            variant["variant_id"] for variant in units[unit_id]["variants"]
        }
        for variant_id in case["variant_ids"]:
            key = (unit_id, variant_id)
            if variant_id not in known_variants:
                raise ValueError(f"{case_id} 引用了未知 variant: {variant_id}")
            covered_variants.add(key)

        relative, qualname = split_nodeid(case["source_test"])
        test_path = pytorch_root / relative
        if not test_path.is_file():
            raise FileNotFoundError(f"community test 文件不存在: {relative}")
        qualnames = qualname_cache.setdefault(test_path, python_qualnames(test_path))
        if qualname not in qualnames:
            raise ValueError(f"community test 方法不存在: {relative}::{qualname}")
        direct_args = case.get("direct_args", [])
        if direct_args:
            if len(direct_args) != len(set(direct_args)):
                raise ValueError(f"{case_id} direct_args 存在重复生成方法名")
            expected_prefix = qualname + "_"
            if any(not value.startswith(expected_prefix) for value in direct_args):
                raise ValueError(
                    f"{case_id} direct_args 必须是 {qualname} 的参数化生成方法名"
                )

    if seen_tests != manifest_tests:
        missing = sorted(manifest_tests - seen_tests)
        extra = sorted(seen_tests - manifest_tests)
        raise ValueError(f"reference cases 未一一覆盖 community tests: {missing=}, {extra=}")

    dispositions: set[tuple[str, str]] = set()
    for item in plan["non_executed_variants"]:
        unit_id = item["acceptance_unit_id"]
        key = (unit_id, item["variant_id"])
        if unit_id not in units:
            raise ValueError(f"non-executed variant 引用了未知 unit: {unit_id}")
        known_variants = {
            variant["variant_id"] for variant in units[unit_id]["variants"]
        }
        if item["variant_id"] not in known_variants:
            raise ValueError(f"non-executed variant 不存在: {key}")
        if key in dispositions or key in covered_variants:
            raise ValueError(f"variant 同时重复映射或排除: {key}")
        dispositions.add(key)

    all_variants = {
        (unit_id, variant["variant_id"])
        for unit_id, unit in units.items()
        for variant in unit["variants"]
    }
    accounted = covered_variants | dispositions
    if accounted != all_variants:
        missing = sorted(all_variants - accounted)
        extra = sorted(accounted - all_variants)
        raise ValueError(f"variant 执行/排除映射不完整: {missing=}, {extra=}")

    actual_commit = git_output(pytorch_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"PyTorch source commit 不匹配: expected={expected_commit}, "
            f"actual={actual_commit}"
        )
    if git_output(pytorch_root, "status", "--short"):
        raise ValueError("PyTorch source working tree 非 clean")
    return {
        "acceptance_units": len(units),
        "cases": len(seen_case_ids),
        "community_tests": len(seen_tests),
        "variants": len(all_variants),
        "executed_variants": len(covered_variants),
        "non_executed_variants": len(dispositions),
    }


def collect_environment(
    pytorch_root: Path,
    work_dir: Path,
    expected_commit: str,
    run_dir: Path,
) -> dict[str, Any]:
    probe = """
import ctypes
import getpass
import importlib.metadata
import json
import os
import platform
import sys
import torch
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON

cuda_driver_api = {
    "library_loaded": False,
    "init_return_code": None,
    "version_return_code": None,
    "version": None,
    "error": None,
}
try:
    libcuda = ctypes.CDLL("libcuda.so.1")
    cuda_driver_api["library_loaded"] = True
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDriverGetVersion.restype = ctypes.c_int
    cuda_driver_api["init_return_code"] = libcuda.cuInit(0)
    driver_version = ctypes.c_int()
    cuda_driver_api["version_return_code"] = libcuda.cuDriverGetVersion(
        ctypes.byref(driver_version)
    )
    if cuda_driver_api["version_return_code"] == 0:
        cuda_driver_api["version"] = driver_version.value
except Exception as error:
    cuda_driver_api["error"] = f"{type(error).__name__}: {error}"

selected_environment = {
    name: os.environ.get(name)
    for name in (
        "HOME",
        "VIRTUAL_ENV",
        "CUDA_HOME",
        "CUDA_COMPAT_DIR",
        "CUDA_VISIBLE_DEVICES",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "PATH",
        "TMPDIR",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    )
}

packages = {}
for name in ("torch", "triton", "pytorch-triton", "pytorch-triton-rocm"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        })

print(json.dumps({
    "user": getpass.getuser(),
    "uid": os.getuid() if hasattr(os, "getuid") else None,
    "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "torch_file": torch.__file__,
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_type": GPU_TYPE,
    "has_gpu_and_triton": HAS_GPU_AND_TRITON,
    "cuda_driver_api": cuda_driver_api,
    "selected_environment": selected_environment,
    "devices": devices,
    "packages": packages,
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = run_command([sys.executable, "-c", probe], cwd=work_dir, env=env)
    (run_dir / "environment_probe_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (run_dir / "environment_probe_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GPU 环境探针失败；详情见 environment_probe_stderr.log"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("GPU 环境探针没有输出 JSON")
    runtime = json.loads(lines[-1])

    source_commit = git_output(pytorch_root, "rev-parse", "HEAD")
    source_status = git_output(pytorch_root, "status", "--short")
    nvidia_smi: dict[str, Any] = {"available": False, "stdout": None, "stderr": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        smi = run_command([executable, "-L"], cwd=work_dir)
        driver_query = run_command(
            [
                executable,
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader",
            ],
            cwd=work_dir,
        )
        smi_version = run_command([executable, "--version"], cwd=work_dir)
        nvidia_smi = {
            "available": True,
            "return_code": smi.returncode,
            "stdout": smi.stdout.strip(),
            "stderr": smi.stderr.strip(),
            "driver_query": {
                "return_code": driver_query.returncode,
                "stdout": driver_query.stdout.strip(),
                "stderr": driver_query.stderr.strip(),
            },
            "version": {
                "return_code": smi_version.returncode,
                "stdout": smi_version.stdout.strip(),
                "stderr": smi_version.stderr.strip(),
            },
        }
    nvcc: dict[str, Any] = {"available": False, "stdout": None, "stderr": None}
    executable = shutil.which("nvcc")
    if executable:
        version = run_command([executable, "--version"], cwd=work_dir)
        nvcc = {
            "available": True,
            "return_code": version.returncode,
            "stdout": version.stdout.strip(),
            "stderr": version.stderr.strip(),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "user": getpass.getuser(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
            "working_directory": str(work_dir),
        },
        "source": {
            "pytorch_root": str(pytorch_root),
            "expected_commit": expected_commit,
            "actual_commit": source_commit,
            "working_tree_state": "clean" if not source_status else "dirty",
            "working_tree_status": source_status.splitlines(),
        },
        "runtime": runtime,
        "nvidia_smi": nvidia_smi,
        "nvcc": nvcc,
    }
    fingerprint_input = json.dumps(
        {key: value for key, value in result.items() if key != "generated_at"},
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    result["fingerprint_sha256"] = sha256_bytes(fingerprint_input)
    write_json(run_dir / "environment.json", result)

    if source_commit != expected_commit:
        raise RuntimeError("PyTorch source commit 与 reference contract 不一致")
    if source_status:
        raise RuntimeError("PyTorch source working tree 非 clean，不能建立 reference baseline")
    if runtime.get("torch_git_version") != expected_commit:
        raise RuntimeError(
            "运行时 torch git_version 与 source/manifest commit 不一致；不能建立 baseline"
        )
    if runtime.get("gpu_type") != "cuda" or not runtime.get("cuda_available"):
        raise RuntimeError("GPU direct reference suite 需要可用 CUDA GPU")
    if not runtime.get("has_gpu_and_triton"):
        raise RuntimeError("PyTorch 测试环境未检测到 GPU+Triton")
    return result


def normalized_digest(content: bytes, roots: list[Path]) -> str:
    text = content.decode("utf-8", errors="replace")
    for root in roots:
        text = text.replace(str(root), "<ROOT>")
    text = re.sub(r"0x[0-9a-fA-F]+", "0x<ADDR>", text)
    text = re.sub(r"run_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d+", "run_<STAMP>", text)
    return sha256_bytes(text.encode("utf-8"))


def collect_fx(
    case_dir: Path,
    debug_root: Path,
    pytorch_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    roots = [case_dir, debug_root, pytorch_root, work_dir]

    def collect(name: str, output_name: str) -> dict[str, Any]:
        paths = sorted(debug_root.rglob(name)) if debug_root.exists() else []
        items: list[dict[str, Any]] = []
        chunks: list[str] = []
        content_digests: list[str] = []
        for index, path in enumerate(paths, start=1):
            content = path.read_bytes()
            digest = normalized_digest(content, roots)
            content_digests.append(digest)
            relative = path.relative_to(case_dir)
            items.append(
                {
                    "path": str(relative),
                    "bytes": len(content),
                    "sha256": sha256_bytes(content),
                    "normalized_sha256": digest,
                }
            )
            chunks.extend(
                [
                    f"===== graph {index}: {relative} =====\n",
                    content.decode("utf-8", errors="replace"),
                    "\n",
                ]
            )
        if not chunks:
            chunks.append("未捕获到对应 FX artifact。\n")
        output = case_dir / output_name
        output.write_text("".join(chunks), encoding="utf-8")
        signature = (
            sha256_bytes("\n".join(sorted(content_digests)).encode("utf-8"))
            if content_digests
            else None
        )
        return {
            "captured": bool(items),
            "count": len(items),
            "combined_path": output.name,
            "files": items,
            "normalized_signature": signature,
        }

    before = collect("fx_graph_readable.py", "fx_before.txt")
    after = collect("fx_graph_transformed.py", "fx_after.txt")
    signature_parts = [
        value
        for value in (
            before["normalized_signature"],
            after["normalized_signature"],
        )
        if value is not None
    ]
    stable_signature = (
        sha256_bytes("\n".join(signature_parts).encode("utf-8"))
        if len(signature_parts) == 2
        else None
    )
    return {"before": before, "after": after, "stable_signature": stable_signature}


def artifact_inventory(case_dir: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_inventory.json", "reference_result.json"}
    inventory = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        content = path.read_bytes()
        inventory.append(
            {
                "path": str(path.relative_to(case_dir)),
                "bytes": len(content),
                "sha256": sha256_bytes(content),
            }
        )
    return inventory


def parse_unittest_output(stdout: str, stderr: str, return_code: int | None) -> dict[str, Any]:
    combined = "\n".join([stdout, stderr])
    ran = re.findall(r"Ran\s+(\d+)\s+tests?", combined)
    skipped = re.findall(r"skipped=(\d+)", combined)
    reasons = re.findall(r"skipped\s+['\"](.+?)['\"]", combined)
    tests_ran = int(ran[-1]) if ran else None
    tests_skipped = int(skipped[-1]) if skipped else 0 if tests_ran else None
    if return_code is None:
        status = "timed-out"
    elif return_code != 0:
        status = "failed"
    elif tests_ran == 0 or tests_ran is None:
        status = "no-tests"
    elif tests_skipped is not None and tests_skipped >= tests_ran:
        status = "skipped"
    else:
        status = "passed"
    return {
        "status": status,
        "success": status == "passed",
        "tests_ran": tests_ran,
        "tests_skipped": tests_skipped,
        "skip_reason": reasons[-1] if reasons else None,
    }


def case_command(
    case: dict[str, Any], repo_root: Path, pytorch_root: Path
) -> list[str]:
    mode = case["tracking_mode"]
    if mode == "direct":
        relative, qualname = split_nodeid(case["source_test"])
        direct_args = case.get("direct_args")
        if direct_args:
            return [sys.executable, str(pytorch_root / relative), "-v", *direct_args]
        return [sys.executable, str(pytorch_root / relative), "-v", qualname]
    entrypoint = (repo_root / case["entrypoint"]).resolve()
    if not entrypoint.is_file() or not is_relative_to(entrypoint, repo_root):
        raise FileNotFoundError(f"adapter/extracted entrypoint 非法: {entrypoint}")
    return [sys.executable, str(entrypoint), "--source-test", case["source_test"]]


def run_case(
    case: dict[str, Any],
    repo_root: Path,
    pytorch_root: Path,
    work_dir: Path,
    run_dir: Path,
    run_id: str,
    expected_commit: str,
    default_timeout: int,
    timeout_override: int | None,
) -> dict[str, Any]:
    case_dir = run_dir / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=False)
    debug_parent = case_dir / "debug"
    trace_dir = case_dir / "structured_trace"
    cache_dir = run_dir / "scratch_cache" / case["case_id"]
    command = case_command(case, repo_root, pytorch_root)
    timeout_seconds = timeout_override or case.get(
        "timeout_seconds", default_timeout
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp(),
        "run_id": run_id,
        "case_id": case["case_id"],
        "acceptance_unit_id": case["acceptance_unit_id"],
        "variant_ids": case["variant_ids"],
        "source_test": case["source_test"],
        "tracking_mode": case["tracking_mode"],
        "expected_match": case["expected_match"],
        "expected_assertions": case["expected_assertions"],
        "pytorch_commit": expected_commit,
    }
    write_json(case_dir / "metadata.json", metadata)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": str(debug_parent),
            "TORCH_TRACE": str(trace_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
        }
    )
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    start = time.monotonic()
    return_code: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            env=env,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    duration = time.monotonic() - start
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_unittest_output(stdout, stderr, return_code)
    fx = collect_fx(case_dir, debug_parent, pytorch_root, work_dir)
    required_artifacts_valid = all(
        fx["before" if item == "fx-before" else "after"]["captured"]
        for item in case["required_artifacts"]
    )
    reference_valid = parsed["success"] and required_artifacts_valid
    if parsed["status"] == "passed":
        assertion_status = (
            "not-directly-asserted-community-test-passed"
            if case["expected_match"] == "not-directly-asserted"
            else "passed-inside-community-test"
        )
        correctness_status = "community-assertion-passed"
    else:
        assertion_status = "not-established"
        correctness_status = "not-established"
    if reference_valid and case["tracking_mode"] == "direct":
        adapter_decision = "not-needed-direct-valid"
    elif reference_valid and case["tracking_mode"] == "adapter":
        adapter_decision = "adapter-valid-after-recorded-direct-blocker"
    elif reference_valid and case["tracking_mode"] == "extracted":
        adapter_decision = "extracted-valid-after-recorded-direct-blocker"
    else:
        adapter_decision = "review-direct-blocker-before-adapter"

    benchmark = {
        "status": "not-configured",
        "functional_gate": "passed" if reference_valid else "not-passed",
        "reason": (
            "本轮仅建立原生 community functional baseline；尚未定义独立 benchmark。"
            if reference_valid
            else "reference functional/artifact gate 未通过，禁止运行 benchmark。"
        ),
    }
    write_json(case_dir / "benchmark.json", benchmark)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp(),
        "run_id": run_id,
        "case": {
            "case_id": case["case_id"],
            "acceptance_unit_id": case["acceptance_unit_id"],
            "variant_ids": case["variant_ids"],
            "source_test": case["source_test"],
            "tracking_mode": case["tracking_mode"],
        },
        "source": {
            "pytorch_root": str(pytorch_root),
            "expected_commit": expected_commit,
            "actual_commit": git_output(pytorch_root, "rev-parse", "HEAD"),
        },
        "environment_fingerprint_ref": "../../environment.json",
        "execution": {
            "status": parsed["status"],
            "success": parsed["success"],
            "return_code": return_code,
            "duration_seconds": round(duration, 6),
            "command": command,
            "working_directory": str(work_dir),
            "timeout_seconds": timeout_seconds,
            "tests_ran": parsed["tests_ran"],
            "tests_skipped": parsed["tests_skipped"],
            "skip_reason": parsed["skip_reason"],
        },
        "match": {
            "expected": case["expected_match"],
            "expected_assertions": case["expected_assertions"],
            "assertion_status": assertion_status,
            "observed_count": None,
            "evidence_method": (
                "原生 community test 内部实际存在的 counter/FileCheck/correctness 断言；"
                "runner 不伪造未打印的原始计数"
            ),
        },
        "fx": fx,
        "correctness": {
            "status": correctness_status,
            "evidence_method": "原生 community test eager/compiled 断言",
        },
        "benchmark": benchmark,
        "adapter_decision": adapter_decision,
        "reference_valid": reference_valid,
    }
    write_json(case_dir / "reference_result.json", result)
    write_json(case_dir / "artifact_inventory.json", artifact_inventory(case_dir))
    return result


def build_variant_summary(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_case = {result["case"]["case_id"]: result for result in results}
    cases_by_variant: dict[tuple[str, str], list[str]] = {}
    for case in plan["cases"]:
        for variant_id in case["variant_ids"]:
            key = (case["acceptance_unit_id"], variant_id)
            cases_by_variant.setdefault(key, []).append(case["case_id"])
    dispositions = {
        (item["acceptance_unit_id"], item["variant_id"]): item
        for item in plan["non_executed_variants"]
    }
    summary = []
    for unit in manifest["acceptance_units"]:
        unit_id = unit["acceptance_unit_id"]
        for variant in unit["variants"]:
            key = (unit_id, variant["variant_id"])
            case_ids = cases_by_variant.get(key, [])
            if key in dispositions:
                item = dispositions[key]
                status = item["disposition"]
                reason = item["reason"]
            elif not case_ids:
                status = "not-selected"
                reason = "本轮过滤未执行关联 case。"
            else:
                selected_results = [
                    results_by_case[case_id]
                    for case_id in case_ids
                    if case_id in results_by_case
                ]
                if len(selected_results) != len(case_ids):
                    status = "not-selected"
                    reason = "本轮只执行了 reference plan 子集。"
                elif all(result["reference_valid"] for result in selected_results):
                    status = "valid-reference"
                    reason = "所有关联原生 community tests 与必需 artifacts 均有效。"
                else:
                    status = "direct-blocked-or-invalid"
                    reason = "至少一个关联原生 community test 或 artifact gate 未通过。"
            summary.append(
                {
                    "acceptance_unit_id": unit_id,
                    "variant_id": variant["variant_id"],
                    "case_ids": case_ids,
                    "status": status,
                    "reason": reason,
                }
            )
    return summary


def write_summary(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    counts = {status: 0 for status in ("passed", "failed", "skipped", "timed-out", "no-tests")}
    for result in results:
        counts[result["execution"]["status"]] += 1
    valid = sum(result["reference_valid"] for result in results)
    variant_summary = build_variant_summary(manifest, plan, results)
    selection_valid = bool(results) and valid == len(results)
    suite_complete = len(selected_cases) == len(plan["cases"])
    suite_valid = suite_complete and selection_valid
    if suite_valid:
        status = "valid-reference-suite"
    elif selection_valid:
        status = "valid-partial-reference-selection"
    else:
        status = "partial-or-invalid-reference-suite"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp(),
        "run_id": run_id,
        "status": status,
        "suite_valid": suite_valid,
        "suite_complete": suite_complete,
        "selection_valid": selection_valid,
        "environment_fingerprint": environment["fingerprint_sha256"],
        "expected_pytorch_commit": manifest["source_baselines"]["pytorch"]["commit"],
        "selected_case_count": len(selected_cases),
        "case_counts": {**counts, "reference_valid": valid},
        "cases": [
            {
                "case_id": result["case"]["case_id"],
                "acceptance_unit_id": result["case"]["acceptance_unit_id"],
                "status": result["execution"]["status"],
                "reference_valid": result["reference_valid"],
                "result": f"cases/{result['case']['case_id']}/reference_result.json",
            }
            for result in results
        ],
        "variants": variant_summary,
        "next_action": (
            "reference 有效；回传 artifacts 后由 NPU 控制节点复核并决定是否冻结 denominator。"
            if suite_valid
            else (
                "选定 case 有效；保留为重跑证据，但完整 suite 未执行，不能冻结 denominator。"
                if selection_valid
                else "先依据 direct artifacts 识别真实 blocker；只有设备/backend/采集阻塞才设计最小 adapter。"
            )
        ),
    }
    write_json(run_dir / "reference_summary.json", summary)
    lines = [
        f"# {plan['task_id']} GPU/reference 执行摘要",
        "",
        f"> 生成时间：{summary['generated_at']}",
        f"> run_id：`{run_id}`",
        f"> 状态：`{summary['status']}`",
        "",
        "## 计数",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| 选择 case | {len(selected_cases)} |",
        f"| reference valid | {valid} |",
        f"| passed | {counts['passed']} |",
        f"| failed | {counts['failed']} |",
        f"| skipped | {counts['skipped']} |",
        f"| timed-out | {counts['timed-out']} |",
        f"| no-tests | {counts['no-tests']} |",
        "",
        "## Case 结果",
        "",
        "| Case | Acceptance unit | 执行状态 | Reference valid |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        case = result["case"]
        lines.append(
            f"| `{case['case_id']}` | `{case['acceptance_unit_id']}` | "
            f"`{result['execution']['status']}` | "
            f"`{str(result['reference_valid']).lower()}` |"
        )
    lines.extend(["", "## 下一步", "", summary["next_action"], ""])
    (run_dir / "reference_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 manifest-driven GPU 原生社区 reference suite"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--pytorch-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("upstream/manifest.yaml"),
        help="相对 repo-root 或绝对 manifest 路径",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=Path("upstream/reference_plan.yaml"),
        help="相对 repo-root 或绝对 reference plan 路径",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    pytorch_root = args.pytorch_root.resolve()
    work_dir = args.work_dir.resolve()
    if not repo_root.is_dir() or not pytorch_root.is_dir() or not work_dir.is_dir():
        raise FileNotFoundError("repo-root、pytorch-root、work-dir 必须存在")
    if is_relative_to(work_dir, pytorch_root):
        raise ValueError("测试工作目录不能位于 PyTorch source tree 内")
    if any(part == "torch_npu" for part in work_dir.parts):
        raise ValueError("测试工作目录不能位于 torch_npu source tree 内")
    manifest_path = args.manifest_path
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    plan_path = args.plan_path
    if not plan_path.is_absolute():
        plan_path = repo_root / plan_path
    manifest = load_json(manifest_path.resolve())
    plan = load_json(plan_path.resolve())
    counts = validate_contract(manifest, plan, pytorch_root)
    print(
        "reference_plan_validation=OK "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    if args.validate_only:
        print("torch_imported=0 gpu_executed=0")
        return 0
    if args.output_root is None:
        raise ValueError("执行 GPU suite 时必须提供 --output-root")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds 必须大于 0")

    cases = plan["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        known = {case["case_id"] for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"未知 case_id: {unknown}")
        cases = [case for case in cases if case["case_id"] in requested]
    run_id = args.run_id or datetime.now().astimezone().strftime(
        "reference-%Y%m%dT%H%M%S%z"
    )
    if not re.fullmatch(r"[A-Za-z0-9._+-]+", run_id):
        raise ValueError("--run-id 只允许字母、数字、点、下划线、加号和连字符")
    output_root = args.output_root.resolve()
    if is_relative_to(output_root, repo_root) or is_relative_to(
        output_root, pytorch_root
    ):
        raise ValueError("原始 reference artifacts 必须写到 tracker/PyTorch 仓库外")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "manifest_snapshot.json", manifest)
    write_json(run_dir / "reference_plan_snapshot.json", plan)
    expected_commit = manifest["source_baselines"]["pytorch"]["commit"]
    try:
        environment = collect_environment(
            pytorch_root, work_dir, expected_commit, run_dir
        )
    except Exception as error:
        write_json(
            run_dir / "reference_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": timestamp(),
                "run_id": run_id,
                "status": "environment-invalid",
                "suite_valid": False,
                "error": str(error),
                "next_action": "修复 GPU/reference 环境后重跑；不得进入 adapter 或 NPU comparison。",
            },
        )
        print(f"reference_environment=INVALID error={error}", file=sys.stderr)
        print(f"artifacts={run_dir}", file=sys.stderr)
        return 2

    results = []
    default_timeout = plan["execution_policy"]["default_timeout_seconds"]
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] START {case['case_id']}", flush=True)
        try:
            result = run_case(
                case,
                repo_root,
                pytorch_root,
                work_dir,
                run_dir,
                run_id,
                expected_commit,
                default_timeout,
                args.timeout_seconds,
            )
        except Exception as error:
            case_dir = run_dir / "cases" / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "runner_error.log").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": timestamp(),
                "run_id": run_id,
                "case": {
                    "case_id": case["case_id"],
                    "acceptance_unit_id": case["acceptance_unit_id"],
                    "variant_ids": case["variant_ids"],
                    "source_test": case["source_test"],
                    "tracking_mode": case["tracking_mode"],
                },
                "source": {
                    "pytorch_root": str(pytorch_root),
                    "expected_commit": expected_commit,
                    "actual_commit": git_output(pytorch_root, "rev-parse", "HEAD"),
                },
                "environment_fingerprint_ref": "../../environment.json",
                "execution": {
                    "status": "failed",
                    "success": False,
                    "return_code": None,
                    "duration_seconds": 0,
                    "command": [],
                    "working_directory": str(work_dir),
                    "timeout_seconds": args.timeout_seconds or default_timeout,
                    "tests_ran": None,
                    "tests_skipped": None,
                    "skip_reason": None,
                },
                "match": {
                    "expected": case["expected_match"],
                    "expected_assertions": case["expected_assertions"],
                    "assertion_status": "runner-error",
                    "observed_count": None,
                    "evidence_method": "runner error before result collection",
                },
                "fx": {
                    "before": {"captured": False, "count": 0},
                    "after": {"captured": False, "count": 0},
                    "stable_signature": None,
                },
                "correctness": {
                    "status": "not-established",
                    "evidence_method": "runner error",
                },
                "benchmark": {
                    "status": "not-configured",
                    "functional_gate": "not-passed",
                    "reason": "runner error",
                },
                "adapter_decision": "review-direct-blocker-before-adapter",
                "reference_valid": False,
            }
            write_json(case_dir / "reference_result.json", result)
        results.append(result)
        print(
            f"[{index}/{len(cases)}] END {case['case_id']} "
            f"status={result['execution']['status']} "
            f"reference_valid={str(result['reference_valid']).lower()}",
            flush=True,
        )

    shutil.rmtree(run_dir / "scratch_cache", ignore_errors=True)
    summary = write_summary(
        run_dir,
        run_id,
        manifest,
        plan,
        cases,
        results,
        environment,
    )
    print(f"reference_suite={summary['status']}")
    print(f"artifacts={run_dir}")
    return 0 if summary["selection_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
