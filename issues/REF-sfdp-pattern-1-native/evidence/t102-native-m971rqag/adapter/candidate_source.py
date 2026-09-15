# Copyright (c) 2026, Huawei Technologies Co., Ltd
"""Build reviewed NPU training patterns using the active decomposition table."""
import functools
import re


def install_sfdp_training_patterns():
    """Called only by triton_experimental activation, before lazy SFDP init."""
    import torch
    from torch._inductor import pattern_matcher as pm
    from torch._inductor.fx_passes import fuse_attention

    original = fuse_attention.gen_register_replacement
    if getattr(original, "_npu_training_patterns", False):
        return

    @functools.wraps(original)
    def register(name, **kwargs):
        inputs = tuple(kwargs["example_inputs"])
        tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
        reviewed = re.fullmatch(r"_sfdp_pattern_[1-5](?:_half)?(?:_bs1)?_training", name)
        if not (reviewed and tensors and all(x.device.type == "npu" for x in tensors)):
            return original(name, **dict(kwargs, example_inputs=inputs))
        # Keep the original replacement, guards, duplicate policy and tracing options.
        # Do not restore FMA or decompose matmul_backward just to match CUDA patterns.
        trace_options = {key: kwargs[key] for key in
                         ("exclusive_arg_names", "get_decomp_fn") if key in kwargs}
        pattern, _ = pm.gen_pattern_and_search_gm(
            kwargs["search_fn"], inputs, kwargs["trace_fn"],
            kwargs.get("scalar_workaround"), **trace_options)
        return pm.register_replacement(
            **dict(kwargs, example_inputs=inputs), pattern_name=name,
            search_fn_pattern=pattern)

    register._npu_training_patterns = True
    fuse_attention.gen_register_replacement = register
