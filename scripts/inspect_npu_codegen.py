#!/usr/bin/env python3
"""只读汇总真实生成代码的调用/设备线索，不自动把空告警判为无fallback。"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def inspect(root):
    calls = Counter()
    suspicious = []
    files = []
    # 只取torch_compile_debug中的实际图，不重复计cache副本。
    paths = sorted(p for p in root.rglob('output_code.py') if 'torch_compile_debug' in p.parts)
    for path in paths:
        text = path.read_text()
        tree = ast.parse(text)
        names = Counter(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call))
        selected = {n:c for n,c in names.items() if n.startswith(('torch.ops.','extern_kernels.')) or n.endswith('.run')}
        calls.update(selected)
        for line_number,line in enumerate(text.splitlines(),1):
            if '.cpu(' in line or "device='cpu'" in line or 'device="cpu"' in line or '.to("cpu")' in line or ".to('cpu')" in line:
                suspicious.append(dict(path=str(path),line=line_number,text=line.strip()))
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calls=selected))
    return dict(code_executed=False,output_code_count=len(files),calls=dict(calls),
                suspicious_cpu_lines=suspicious,files=files,verdict='manual-review-required')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    record=inspect(args.run_dir.resolve())
    text=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        if args.output.exists():
            raise ValueError('检查记录必须写入新文件，不覆盖旧审计')
        args.output.write_text(text)
    print(json.dumps({k:v for k,v in record.items() if k!='files'},ensure_ascii=False,indent=2))
