"""GPU 共享/独占/排队的零设备测试；使用假的 nvidia-smi，不接触真实 GPU。"""

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")


class GPUFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-wait-test-", dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        smi = self.bin / "nvidia-smi"
        smi.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
kind = 'info' if any(a.startswith('--query-gpu=') for a in sys.argv) else 'processes'
root = Path(os.environ['GPU_WAIT_FIXTURE'])
counter = root / (kind + '.count')
index = int((counter.read_text() if counter.exists() else '0') or '0')
counter.write_text(str(index + 1))
sequence = json.loads(os.environ['GPU_WAIT_' + kind.upper()])
item = sequence[min(index, len(sequence) - 1)]
print(item['output'])
sys.exit(item.get('code', 0))
''')
        smi.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", GPU_WAIT_FIXTURE=str(self.root),
                        GPU_WAIT_INFO=json.dumps([{"output": "Default, 60000"}]),
                        GPU_WAIT_PROCESSES=json.dumps([{"output": "12345"}]))


class GPUWaitTests(GPUFixtures):
    def command(self, mode="shared", wait=0, timeout=0, poll=1, minimum=1024, fake_sleep=True, hold=False, gpu="2"):
        code = 'set -euo pipefail\nsource "$1"\n'
        if fake_sleep:
            code += 'tracker_gpu_wait_sleep() { SECONDS=$((SECONDS + $1)); }\n'
        code += 'tracker_gpu_acquire "$2" "$3" "$4" "$5" "$6" "$7" "$8"\n'
        code += 'echo "fixture_started mode=${PASS_GPU_EXECUTION_MODE} compute=${PASS_GPU_COMPUTE_MODE}"\n'
        if hold:
            code += 'read -r fixture_release\n'
        return ["bash", "-c", code, "gpu-wait-fixture", str(ROOT / "scripts/gpu_wait.sh"),
                str(gpu), str(wait), str(timeout), str(poll), str(self.root / "locks"), mode, str(minimum)]

    def run_helper(self, **kwargs):
        return subprocess.run(self.command(**kwargs), env=self.env, cwd=WORK, capture_output=True, text=True, timeout=10)

    def test_shared_default_accepts_existing_compute_processes(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("execution_mode=shared", result.stdout)
        self.assertFalse((self.root / "processes.count").exists())

    def test_exclusive_busy_without_wait_exits_three(self):
        result = self.run_helper(mode="exclusive")
        self.assertEqual(result.returncode, 3)
        self.assertNotIn("fixture_started", result.stdout)

    def test_exclusive_waits_until_processes_exit(self):
        self.env["GPU_WAIT_PROCESSES"] = json.dumps([{"output": "12345"}, {"output": ""}])
        result = self.run_helper(mode="exclusive", wait=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reason=compute-processes", result.stdout)
        self.assertRegex(result.stdout, r"waited_seconds=[1-9][0-9]*")

    def test_shared_waits_for_memory_not_idle_gpu(self):
        self.env["GPU_WAIT_INFO"] = json.dumps([{"output": "Default, 500"}, {"output": "Default, 1024"}])
        result = self.run_helper(wait=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reason=insufficient-memory", result.stdout)
        self.assertFalse((self.root / "processes.count").exists())

    def test_zero_memory_floor_can_be_selected(self):
        self.env["GPU_WAIT_INFO"] = json.dumps([{"output": "Default, 0"}])
        self.assertEqual(self.run_helper().returncode, 3)
        self.assertEqual(self.run_helper(minimum=0).returncode, 0)

    def test_wait_timeout_never_starts_and_caps_last_delay(self):
        result = self.run_helper(mode="exclusive", wait=1, timeout=2, poll=30)
        self.assertEqual(result.returncode, 124)
        self.assertRegex(result.stdout, r"next_check_seconds=[12]\b")
        self.assertNotIn("fixture_started", result.stdout)

    def test_query_error_does_not_wait_forever_or_start(self):
        self.env["GPU_WAIT_INFO"] = json.dumps([{"output": "driver error", "code": 9}])
        result = self.run_helper(wait=1)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("fixture_started", result.stdout)

    def test_unrecognized_query_or_prohibited_mode_is_error(self):
        for output in ("Default, N/A", "Prohibited, 60000", "Default, 60000\nDefault, 70000"):
            with self.subTest(output=output):
                self.env["GPU_WAIT_INFO"] = json.dumps([{"output": output}])
                self.assertEqual(self.run_helper(wait=1).returncode, 2)

    def test_hardware_exclusive_mode_cannot_be_bypassed_by_shared(self):
        self.env["GPU_WAIT_INFO"] = json.dumps([{"output": "Exclusive_Process, 60000"}])
        self.assertEqual(self.run_helper().returncode, 3)
        self.env["GPU_WAIT_PROCESSES"] = json.dumps([{"output": ""}])
        self.assertEqual(self.run_helper().returncode, 0)

    def test_bad_options_rejected_before_gpu_query(self):
        for kwargs in ({"poll": 0}, {"poll": 61}, {"timeout": -1}, {"minimum": "1.5"}, {"gpu": "2,3"}, {"mode": "any"}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_helper(**kwargs).returncode, 2)
        self.assertFalse((self.root / "info.count").exists())

    def test_leading_zero_gpu_uses_same_lock(self):
        result = self.run_helper(gpu="02")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "locks/gpu-2.lock").is_file())
        self.assertFalse((self.root / "locks/gpu-02.lock").exists())

    def test_symlink_lock_is_rejected(self):
        locks = self.root / "locks"
        locks.mkdir()
        target = self.root / "original"
        target.write_text("preserve")
        (locks / "gpu-2.lock").symlink_to(target)
        self.assertEqual(self.run_helper().returncode, 2)
        self.assertEqual(target.read_text(), "preserve")

    def start_helper(self, marker, **kwargs):
        process = subprocess.Popen(self.command(**kwargs), env=self.env, cwd=WORK, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        self.addCleanup(self.close_process, process)
        collected = b""
        deadline = time.monotonic() + 5
        while marker.encode() not in collected and time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                block = os.read(process.stdout.fileno(), 4096)
                if not block:
                    break
                collected += block
        self.assertIn(marker.encode(), collected)
        return process

    @staticmethod
    def close_process(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()

    def test_shared_lock_allows_peer_but_blocks_exclusive(self):
        holder = self.start_helper("fixture_started", hold=True)
        self.assertEqual(self.run_helper().returncode, 0)
        blocked = self.run_helper(mode="exclusive")
        self.assertEqual(blocked.returncode, 3)
        self.assertIn("tracker-lock", blocked.stderr)
        holder.stdin.write(b"done\n")
        holder.stdin.flush()
        self.assertEqual(holder.wait(timeout=3), 0)
        self.env["GPU_WAIT_PROCESSES"] = json.dumps([{"output": ""}])
        self.assertEqual(self.run_helper(mode="exclusive").returncode, 0)

    def test_exclusive_lock_blocks_shared_peer(self):
        self.env["GPU_WAIT_PROCESSES"] = json.dumps([{"output": ""}])
        self.start_helper("fixture_started", mode="exclusive", hold=True)
        self.assertEqual(self.run_helper().returncode, 3)

    def test_interrupt_cancels_wait_without_starting(self):
        process = self.start_helper("gpu_wait=queued", mode="exclusive", wait=1, poll=30, fake_sleep=False)
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 130, output)
        self.assertNotIn(b"fixture_started", output)
        self.env["GPU_WAIT_PROCESSES"] = json.dumps([{"output": ""}])
        self.assertEqual(self.run_helper(mode="exclusive").returncode, 0)

    def test_launcher_rejects_bad_wait_flags_before_environment_setup(self):
        for args in (["--wait-timeout"], ["--wait-timeout", "10"], ["--wait-gpu", "--poll-interval", "0"]):
            with self.subTest(args=args):
                result = subprocess.run(["bash", str(ROOT / "scripts/run_gpu_reference_task.sh"), "--task", "T-078", *args],
                                        cwd=WORK, env=dict(self.env, TRACKER_ROOT=str(ROOT)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)


class GPULauncherTests(GPUFixtures):
    def setUp(self):
        super().setUp()
        # 启动目录仍为 WORK；只将本测试的 mktemp 日志落到 fixture 内，随临时目录回收。
        mktemp = self.bin / "mktemp"
        mktemp.write_text(f"#!{sys.executable}\n" + '''import os, tempfile
fd, path = tempfile.mkstemp(prefix='launcher-', suffix='.log', dir=os.environ['GPU_WAIT_FIXTURE'])
os.close(fd)
print(path)
''')
        mktemp.chmod(0o755)
        self.data = self.root / "data"
        tracker = self.root / "tracker"
        scripts = tracker / "scripts"
        scripts.mkdir(parents=True)
        for name in ("gpu_wait.sh", "export_reference_text.py", "publish_reference_latest.py"):
            (scripts / name).symlink_to(ROOT / "scripts" / name)
        files = (
            "src/pytorch/torch/__init__.py", "envs/PassGPURef/bin/activate", "cuda-12.6/bin/nvcc",
            "envs/PassGPURef/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h",
            "envs/PassGPURef/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
        )
        for relative in files:
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        (self.data / "envs/PassGPURef/bin/python").symlink_to(sys.executable)
        (scripts / "run_t078_reference_all.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == --validate-only ]]; then echo fixture_static=passed; exit 0; fi
done
"${PYTHON}" "${GPU_WAIT_FIXTURE}/fake_runner.py" "$@"
''')
        (self.root / "fake_runner.py").write_text('''import json, os, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--output-root') + 1]) / 'reference-fixture'
root.mkdir()
keys = ('PASS_GPU_EXECUTION_MODE', 'PASS_GPU_COMPUTE_MODE', 'PASS_GPU_MIN_FREE_MEMORY_MIB', 'CUDA_VISIBLE_DEVICES')
environment = {'runtime': {'selected_environment': {key: os.environ.get(key) for key in keys}}, 'cwd': os.getcwd()}
for name, value in {'environment': environment, 'reference_summary': {'run_id': root.name, 'cases': []}, 'manifest_snapshot': {}, 'reference_plan_snapshot': {}}.items():
    (root / (name + '.json')).write_text(json.dumps(value))
print('artifacts=' + str(root))
''')
        self.env.update(TRACKER_ROOT=str(tracker), PASS_GPU_DATA_ROOT=str(self.data),
                        PASS_TRACKER_WORK_DIR=str(WORK), PYTORCH_ROOT=str(self.data / "src/pytorch"),
                        PASS_GPU_VENV=str(self.data / "envs/PassGPURef"), CUDA_HOME=str(self.data / "cuda-12.6"))

    def run_launcher(self, *args):
        return subprocess.run(["bash", str(ROOT / "scripts/run_gpu_reference_task.sh"), "--task", "T-078", "--gpu", "2", *args],
                              env=self.env, cwd=WORK, capture_output=True, text=True, timeout=10)

    def test_shared_default_reaches_runner_and_exports_mode(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads((self.data / "tmp/t078-reference-results/latest-text-handoff.json").read_text())
        env = payload["environment"]["runtime"]["selected_environment"]
        self.assertEqual(env["PASS_GPU_EXECUTION_MODE"], "shared")
        self.assertEqual(env["PASS_GPU_COMPUTE_MODE"], "DEFAULT")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(payload["environment"]["cwd"], str(WORK))
        self.assertEqual(payload["handoff_format_version"], "1.1")
        self.assertEqual(payload["raw_text_transfer"]["omitted_files"], [])
        self.assertEqual(len(payload["raw_text_files"]), 4)

    def test_exclusive_flag_blocks_busy_gpu_before_runner(self):
        result = self.run_launcher("--exclusive")
        self.assertEqual(result.returncode, 3)
        self.assertFalse((self.data / "tmp/t078-reference-results/latest").exists())

    def test_validate_only_never_queries_or_waits_for_gpu(self):
        result = self.run_launcher("--exclusive", "--wait-gpu", "--validate-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "info.count").exists())

    def test_aggressive_shared_wait_continues_when_memory_is_available(self):
        self.env["GPU_WAIT_INFO"] = json.dumps([{"output": "Default, 0"}, {"output": "Default, 60000"}])
        result = self.run_launcher("--wait-gpu", "--wait-timeout", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("next_check_seconds=1", result.stdout)
        self.assertFalse((self.root / "processes.count").exists())


if __name__ == "__main__":
    unittest.main()
