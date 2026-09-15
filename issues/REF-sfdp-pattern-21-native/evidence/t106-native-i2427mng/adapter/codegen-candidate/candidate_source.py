"""隔离codegen候选：每个load保留真实广播形状，select拓宽加载显式限制边界。"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import textwrap


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('候选源码锚点变化，拒绝猜测应用')
    return source.replace(before, after)


def candidate_sources(record_source, rewrite_source):
    record_source = textwrap.dedent(record_source)
    rewrite_source = textwrap.dedent(rewrite_source)
    record_source = replace_once(record_source,
        '    self._npu_select_lane_loads[var_name] = {',
        '''    # 仅对可证明的零offset静态存储范围拓宽；未知布局保留原load。
    storage_size = None
    try:
        layout = V.graph.get_buffer(name).get_layout()
        extent = layout.storage_size()
        if layout.offset == 0 and isinstance(extent, (int, sympy.Integer)) and extent > 0:
            storage_size = int(extent)
    except (AttributeError, KeyError, NotImplementedError):
        pass
    self._npu_select_lane_loads[var_name] = {''')
    record_source = replace_once(record_source,
        '        "lane": k,', '        "lane": k,\n        "storage_size": storage_size,')
    rewrite_source = replace_once(rewrite_source,
        '        lane_var = f"_es_lane{counter}"',
        '''        extent = target.get("storage_size")
        pointer = load_ast.args[0]
        # 只处理可以直接分离基指针和整数offset的普通加载。
        if (extent is None or not isinstance(pointer, ast.BinOp)
                or not isinstance(pointer.op, ast.Add)
                or ast.unparse(pointer.left) != target["pointer"]):
            new_lines.append(line)
            continue
        offset = ast.unparse(pointer.right)
        lane_var = f"_es_lane{counter}"''')
    rewrite_source = replace_once(rewrite_source,
        '        ast.fix_missing_locations(value_ast)',
        '''        # 偏移1等尾部load不能读取多出的lane；首lane仍保留原mask语义。
        bound = ast.parse(
            f"(({offset} + {lane_var}) >= 0) & (({offset} + {lane_var}) < {extent})",
            mode="eval",
        ).body
        mask_kw = next((kw for kw in load_ast.keywords if kw.arg == "mask"), None)
        if len(load_ast.args) > 1:
            load_ast.args[1] = ast.BinOp(left=load_ast.args[1], op=ast.BitAnd(), right=bound)
        elif mask_kw is not None:
            mask_kw.value = ast.BinOp(left=mask_kw.value, op=ast.BitAnd(), right=bound)
        else:
            load_ast.keywords.append(ast.keyword(arg="mask", value=bound))
        ast.fix_missing_locations(value_ast)''')
    rewrite_source = replace_once(rewrite_source,
        '            "[" + ", ".join(sizes) + "]",',
        '''            "[" + ", ".join("1" if s == inner_slot else f"{full_var}.shape[{s}]"
                             for s in range(ndim)) + "]",''')
    return record_source, rewrite_source


def install(output: Path):
    """仅替换本进程class方法；不写site-packages、不改变后端或关闭优化。"""
    from torch_npu._inductor.triton_experimental.codegen import triton as module
    cls = module.NPUTritonKernel
    names = ('_maybe_record_select_lane_load', '_maybe_rewrite_select_lane_load')
    originals = [inspect.getsource(getattr(cls, name)) for name in names]
    updated = candidate_sources(*originals)
    output.mkdir(parents=True, exist_ok=False)
    source_file = Path(module.__file__).resolve()
    (output/'installed_triton_source.py').write_bytes(source_file.read_bytes())
    (output/'candidate_source.py').write_bytes(Path(__file__).read_bytes())
    for name, before, after in zip(names, originals, updated):
        (output/f'before-{name}.py').write_text(textwrap.dedent(before))
        path = output/f'after-{name}.py'
        path.write_text(after)
        namespace = dict(vars(module))
        exec(compile(after, str(path), 'exec'), namespace)
        setattr(cls, name, namespace[name])
    return dict(status='isolated-codegen-candidate-not-deployed',
                installed_source_path=str(source_file),
                installed_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_snapshot=str(output/'installed_triton_source.py'),
                scope='select-only per-load shape and bounded extra lanes; strided-slice branch unchanged')
