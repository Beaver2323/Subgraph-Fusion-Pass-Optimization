"""仅供显式进程局部验证的 NPU 候选，未接入产品启动或修改上游 guard。"""


def install():
    import torch
    from torch._inductor.fx_passes.post_grad import pass_patterns
    from torch._inductor.pattern_matcher import fwd_only, register_replacement
    from torch_npu.utils._dynamo import _InductorNpuRegistry

    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    # 同冻结 misc_patterns.py:173-213 的 pre-SM100 算法；不调用 NVIDIA PTX。
    # 合同只声明社区已覆盖的正、正规 FP32 输入，不外推到负数/NaN/非正规值。
    def e8m0_rceil_log2_pattern(inp):
        return (torch.clamp(torch.ceil(torch.log2(inp)), -127, 127) + 127).to(torch.uint8)

    def e8m0_rceil_log2_replacement(inp):
        inp_bits = inp.view(torch.int32)
        biased_exp = (inp_bits >> 23) & 0xFF
        mantissa = inp_bits & 0x7FFFFF
        needs_round_up = mantissa != 0
        e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
        return torch.clamp(e8m0_biased, 0, 254).to(torch.uint8)

    def e8m0_extra_check(match):
        inp = match.kwargs.get('inp')
        value = inp.meta.get('val') if inp is not None else None
        return value is not None and value.device.type == 'npu' and value.dtype == torch.float32

    register_replacement(e8m0_rceil_log2_pattern, e8m0_rceil_log2_replacement,
        [torch.randn(32, device='npu', dtype=torch.float32).abs() + 1e-10], fwd_only,
        [pass_patterns[1]], extra_check=e8m0_extra_check, skip_duplicates=True,
        pattern_name='e8m0_rceil_log2_pattern')
