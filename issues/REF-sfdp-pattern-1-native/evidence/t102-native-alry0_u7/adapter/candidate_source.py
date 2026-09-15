"""隔离候选：按实际 NPU decomposition 生成指定 SFDP 的训练注册；不修改安装包。"""
from pathlib import Path


def install(pattern: int, output: Path):
    import torch
    from torch._inductor import pattern_matcher as pm
    from torch._inductor.fx_passes import fuse_attention

    original = fuse_attention.gen_register_replacement
    records = []

    def register(name, **kwargs):
        inputs = tuple(kwargs['example_inputs'])
        npu = any(isinstance(x, torch.Tensor) and x.device.type == 'npu' for x in inputs)
        if not (npu and name.startswith(f'_sfdp_pattern_{pattern}_') and name.endswith('_training')):
            return original(name, **kwargs)
        kwargs = dict(kwargs, example_inputs=inputs)
        pat, gm = pm.gen_pattern_and_search_gm(
            kwargs['search_fn'], inputs, kwargs['trace_fn'], kwargs.get('scalar_workaround'))
        output.mkdir(parents=True, exist_ok=True)
        (output / f'target-{name}.py').write_text(gm.code)
        records.append({'pattern_name': name, 'device': 'npu',
                        'mode': 'runtime-decomposition-training-registration',
                        'product_gate_bypassed': False})
        return pm.register_replacement(**kwargs, pattern_name=name, search_fn_pattern=pat)

    fuse_attention.gen_register_replacement = register
    return records
