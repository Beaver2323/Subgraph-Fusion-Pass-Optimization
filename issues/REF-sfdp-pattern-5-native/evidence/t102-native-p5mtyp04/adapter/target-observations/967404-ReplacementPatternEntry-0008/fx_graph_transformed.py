


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    full_default = torch.ops.aten.full.default([16, 16], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  full_default = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(primals_1, primals_2, primals_3, 2, 'BNSD', None, None, None, 0.17677669529663687)
    getitem_4 = npu_fusion_attention_v3_default[0]
    getitem_5 = npu_fusion_attention_v3_default[1]
    getitem_6 = npu_fusion_attention_v3_default[2]
    getitem_7 = npu_fusion_attention_v3_default[3]
    getitem_8 = npu_fusion_attention_v3_default[4]
    getitem_9 = npu_fusion_attention_v3_default[5];  npu_fusion_attention_v3_default = None
    npu_fusion_attention_grad_v3_default = torch.ops.npu.npu_fusion_attention_grad_v3.default(primals_1, primals_2, primals_3, tangents_1, 2, 'BNSD', softmax_max = getitem_5, softmax_sum = getitem_6, softmax_in = getitem_7, attention_in = getitem_4, scale_value = 0.17677669529663687, seed = getitem_8, offset = getitem_9);  primals_1 = primals_2 = primals_3 = tangents_1 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = None
    getitem_10 = npu_fusion_attention_grad_v3_default[0]
    getitem_11 = npu_fusion_attention_grad_v3_default[1]
    getitem_12 = npu_fusion_attention_grad_v3_default[2];  npu_fusion_attention_grad_v3_default = None
    return pytree.tree_unflatten([getitem_4, getitem_10, getitem_11, getitem_12], self._out_spec)
    