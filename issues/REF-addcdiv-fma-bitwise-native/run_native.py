#!/usr/bin/env python3
"""T-078 首个社区方法原生入口预检；不修改测试体、device guard 或产品配置。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


WORK = Path('/home/z50063656/tmp')
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py')
METHOD = 'GPUTests.test_addcdiv_fma_bitwise_equal_cuda'
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--adapter', action='store_true', help='仅在已记录原生入口阻断后调用同目录最小适配器')
mode.add_argument('--source-fix', action='store_true', help='加载 torch_npu 工作树正式修复后运行同一社区方法')
args = parser.parse_args()
if Path.cwd().resolve() != WORK:
    raise SystemExit('必须从 /home/z50063656/tmp 启动')
root = WORK / 't078-npu-results' / 'REF-addcdiv-fma-bitwise-native'
root.mkdir(parents=True, exist_ok=True)
prefix = 'source-fix-' if args.source_fix else ('adapter-' if args.adapter else 'direct-')
run = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
env = dict(os.environ, TORCHINDUCTOR_NPU_BACKEND='triton_experimental',
           TORCH_DEVICE_BACKEND_AUTOLOAD='1', TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',
           TORCHINDUCTOR_COMPILE_THREADS='1', TORCH_COMPILE_DEBUG='1',
           TORCH_COMPILE_DEBUG_DIR=str(run / 'debug'), TORCH_TRACE=str(run / 'trace'),
           TORCHINDUCTOR_CACHE_DIR=str(run / 'inductor-cache'), TRITON_CACHE_DIR=str(run / 'triton-cache'))
if args.source_fix:
    overlay = Path(__file__).parents[2] / "runners" / "t078_source_overlay"
    previous_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{overlay}{os.pathsep}{previous_pythonpath}' if previous_pythonpath else str(overlay)
code = '''import json, os, runpy, sys
from pathlib import Path
import torch, torch_npu
from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
register_inductor_npu()
metadata = {'backend': _InductorNpuRegistry._loaded_backend, 'torch': torch.__version__,
            'torch_npu': torch_npu.__version__, 'device_available': torch.npu.is_available(),
            'torch_commit': torch.version.git_version, 'cwd': os.getcwd(),
            'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}
Path(sys.argv[3]).write_text(json.dumps(metadata, indent=2) + '\\n')
assert metadata['backend'] == 'triton_experimental'
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
command = [sys.executable, '-c', code, str(SOURCE), METHOD, str(run / 'environment.json')]
if args.adapter:
    command = [sys.executable, str(Path(__file__).with_name('npu_adapter.py')), '--artifact-dir', str(run)]
elif args.source_fix:
    command = [sys.executable, str(Path(__file__).with_name('npu_adapter.py')), '--artifact-dir', str(run), '--source-fix']
with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as err:
    try:
        result = subprocess.run(command, cwd=WORK, env=env, stdout=out, stderr=err, timeout=180)
        status = result.returncode
    except subprocess.TimeoutExpired:
        status = 124
summary = {'source_test': str(SOURCE) + '::' + METHOD, 'return_code': status,
           'tracking_mode': 'source-fix-verification' if args.source_fix else ('adapter' if args.adapter else 'direct'), 'raw_logs': str(run), 'body_or_assertions_modified': False,
           'reference_raw_evidence_pending': True}
(run / 'direct_result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False), flush=True)
raise SystemExit(status)
