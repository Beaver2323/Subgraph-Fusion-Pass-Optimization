"""性能证据归档范围与历史保护；不导入torch，不执行设备。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('attention_archive', ROOT/'scripts/review_attention_functional_gate.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class AttentionArchiveTests(unittest.TestCase):
    def test_explicit_installed_reference_needs_verified_deployment(self):
        import validate_attention_slice_deployment as deployment
        with patch.object(deployment, 'verify', return_value={'installed_passed': False}):
            with self.assertRaisesRegex(ValueError, '部署回归未完成'):
                archive.community_for_gate({'deployment': {'path': 'fixture'}}, 22, Path('/missing'))

    def test_old_baseline_reference_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root/'old.json'; p.write_text('{}')
            stage = {'baseline': {'path': p.name, 'sha256': archive.sha(p)}}
            with patch.object(archive, 'ROOT', root):
                self.assertEqual(archive.community_for_gate(stage, 22), p)

    def test_isolated_or_incomplete_explicit_community_cannot_sign(self):
        import validate_attention_slice_deployment as deployment
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root/'issues/REF-sfdp-pattern-22-native/evidence/run/adapter/result.json'
            p.parent.mkdir(parents=True); p.write_text(json.dumps({'status': 'community-contract-passed'}))
            with patch.object(archive, 'ROOT', root), patch.object(deployment, 'verify',
                    return_value={'installed_passed': True}):
                with self.assertRaisesRegex(ValueError, '无候选原方法覆盖'):
                    archive.community_for_gate({'deployment': {'path': 'fixture'}}, 22, p)

    def test_excludes_cache_and_trace_preserves_actual_debug_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ('debug/torch_compile_debug/run/model/output_code.py', 'result.json',
                     'snapshot-worker.py', 'target-1/before.py', 'inductor-cache/output_code.py',
                     'trace/result.json', 'kernel.so')
            for name in names:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# unit-test fixture\n')
            selected = {str(p.relative_to(root)) for p in archive.selected_files(root)}
            self.assertEqual(selected, set(names[:3]))

    def test_rejects_overwrite_and_missing_codegen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            destination = root/'destination'
            with self.assertRaisesRegex(ValueError, '生成代码'):
                archive.archive(source, destination)
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, '覆盖'):
                archive.archive(source, destination)

    def test_archive_bytes_unchanged_and_inventory_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            (source/'output_code.py').write_bytes(b'# exact source\n')
            destination = root/'archive'
            with patch.object(archive, 'ROOT', root):
                archive.archive(source, destination)
            self.assertEqual((source/'output_code.py').read_bytes(), (destination/'output_code.py').read_bytes())
            record = archive.read(destination/'inventory.json')['files'][0]
            self.assertEqual(record['path'], 'archive/output_code.py')
            self.assertEqual(record['sha256'], archive.sha(source/'output_code.py'))
