"""超时清理只能针对本次创建的进程组，不能扫描/终止其他设备用户。"""
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch, call

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_prepared_performance import run_arm


class ProcessIsolationTests(unittest.TestCase):
    def test_fresh_process_group_and_explicit_environment(self):
        child=Mock(pid=54321)
        child.wait.return_value=0
        env={'TORCHINDUCTOR_NPU_BACKEND':'triton_experimental'}
        with patch('run_prepared_performance.subprocess.Popen',return_value=child) as spawn:
            self.assertEqual(run_arm(['python','case.py'],'/home/z50063656/tmp',None,None,env=env),0)
        spawn.assert_called_once_with(['python','case.py'],cwd='/home/z50063656/tmp',
            stdout=None,stderr=None,start_new_session=True,env=env)

    def test_timeout_sends_term_only_to_created_group(self):
        child=Mock(pid=54321)
        child.wait.side_effect=[subprocess.TimeoutExpired(['python'],1),0]
        with patch('run_prepared_performance.subprocess.Popen',return_value=child), \
             patch('run_prepared_performance.os.killpg') as kill, \
             self.assertRaises(subprocess.TimeoutExpired):
            run_arm(['python'],'/home/z50063656/tmp',None,None,timeout=1)
        kill.assert_called_once_with(54321,signal.SIGTERM)

    def test_stuck_child_gets_kill_in_same_group(self):
        child=Mock(pid=54321)
        child.wait.side_effect=[subprocess.TimeoutExpired(['python'],1),
                               subprocess.TimeoutExpired(['python'],5),-9]
        with patch('run_prepared_performance.subprocess.Popen',return_value=child), \
             patch('run_prepared_performance.os.killpg') as kill, \
             self.assertRaises(subprocess.TimeoutExpired):
            run_arm(['python'],'/home/z50063656/tmp',None,None,timeout=1)
        self.assertEqual(kill.call_args_list,[call(54321,signal.SIGTERM),call(54321,signal.SIGKILL)])
