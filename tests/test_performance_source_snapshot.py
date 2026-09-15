import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class SourceSnapshotTests(unittest.TestCase):
    def test_namespace_and_no_file_modules_are_ignored(self):
        for filename in ('t091_t100_performance_worker.py','t102_t107_attention_performance_worker.py'):
            self.check_worker(filename)

    def check_worker(self,filename):
        path = Path(__file__).resolve().parents[1]/'runners'/filename
        spec = importlib.util.spec_from_file_location('worker_source_snapshot_test',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = types.ModuleType('triton.namespace')
        namespace.__file__ = None
        known = types.ModuleType('torch._inductor.fake_for_test')
        known.__file__ = str(path)
        with patch.dict(sys.modules, {namespace.__name__:namespace,known.__name__:known}):
            result = module.source_hashes()
        self.assertEqual(result[str(path)],module.sha256(path))
