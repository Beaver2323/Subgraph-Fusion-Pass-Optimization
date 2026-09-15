#!/usr/bin/env python3
"""只读复核 select-load 的完整设备边界原件；不导入 torch，不执行归档代码。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
CANDIDATE = '6222c9e872f94d4366db4b5dc88f659c54723cfa5bdcc67321c4d14f62650c1e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, product_sha, runner_sha, arm):
    path = path.resolve(strict=True)
    root = path.parent
    record = json.loads(path.read_text())
    require(arm in ('candidate', 'installed'), '只能评审候选或无候选安装态')
    require(record['status'] == 'passed' and record['arm'] == arm
            and record['device_execution'] is True and record['target_exercised'] is True,
            '没有完整通过的实际设备边界')
    require(record['backend'] == 'triton_experimental' and record['pytorch_commit'] == COMMIT,
            '后端或冻结源码不符')
    require(record['installed_before'] == record['installed_after'] == product_sha
            == sha(root / 'installed_triton_source.py'), '安装态源码在执行中变化或快照不符')
    require(record['source_sha256'] == runner_sha == sha(root / 'runner_source.py'),
            '边界执行器快照不符')
    if arm == 'candidate':
        require(record['candidate']['candidate_sha256'] == CANDIDATE
                == sha(root / 'candidate/candidate_source.py'), '不是已复核的同一候选')
        require(record['candidate']['installed_source_sha256'] == product_sha
                == sha(root / 'candidate/installed_triton_source.py'), '候选基线与安装态不符')
    else:
        require(record['candidate'] is None, '安装态复验不得加载候选')
    expected = [(d, w, lane, 'shared', False)
                for d in ('float32', 'float16', 'bfloat16')
                for w in (2, 4, 8) for lane in (0, w-1)]
    expected += [('float32', 4, 3, layout, False) for layout in ('offset', 'strided', 'full')]
    expected += [('float32', 4, 3, 'shared', True)]
    require(len(record['cases']) == len(expected), '边界场景缺失或重复')
    executions = 0
    for index, (row, contract) in enumerate(zip(record['cases'], expected)):
        require(row['index'] == index and row['status'] == 'passed'
                and tuple(row[k] for k in ('dtype','width','lane','layout','dynamic')) == contract,
                '边界场景身份或结果不符')
        rows = [7, 11] if row['dynamic'] else [7]
        require([e['rows'] for e in row['executions']] == rows, '动态第二形状等执行缺失')
        for e in row['executions']:
            require(e['bitwise_equal'] is True and e['mismatch'] == 0
                    and e['output_shape'] == [3, e['rows'], 5], '实际数值/形状断言不通过')
            require(e['storage_offset'] == (1 if row['layout'] == 'offset' else 0),
                    'offset边界没有按合同执行')
            require(e['stride'][-1] == (2 if row['layout'] == 'strided' else 1),
                    'stride边界没有按合同执行')
            executions += 1
    observed = 0
    for item in record['emitted']:
        file = root / item['file']
        require(file.resolve().is_relative_to(root) and sha(file) == item['sha256'],
                '发射代码缺失、越界或哈希不符')
        source = file.read_text()
        require(0 <= item['case'] < 22, '发射代码关联了未知场景')
        if item['per_load_shape'] is True and item['bounded_load'] is True:
            require('_es_full' in source and '.shape[' in source and '>= 0' in source,
                    '候选关键发射没有实际证据')
            observed += 1
    require(observed > 0 and executions == 23, '修复路径未实际执行或执行数不完整')
    return dict(status='boundary-reviewed', arm=arm, scenarios=22, executions=executions,
                bounded_per_load_emissions=observed, result_sha256=sha(path),
                product_sha256=product_sha, runner_sha256=runner_sha,
                performance_gate_issued=False, archive_code_executed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--product-sha256', required=True)
    p.add_argument('--runner-sha256', required=True)
    p.add_argument('--arm', choices=('candidate', 'installed'), required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.result, a.product_sha256, a.runner_sha256, a.arm),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
