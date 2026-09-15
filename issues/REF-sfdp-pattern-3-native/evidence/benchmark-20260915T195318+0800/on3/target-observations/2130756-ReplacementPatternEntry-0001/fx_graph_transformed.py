


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(primals_2, primals_1, primals_3, 4, 'BNSD', None, None, None, 0.5, 0.99999999999)
    getitem_6 = npu_fusion_attention_v3_default[0]
    getitem_7 = npu_fusion_attention_v3_default[1]
    getitem_8 = npu_fusion_attention_v3_default[2]
    getitem_9 = npu_fusion_attention_v3_default[3]
    getitem_10 = npu_fusion_attention_v3_default[4]
    getitem_11 = npu_fusion_attention_v3_default[5];  npu_fusion_attention_v3_default = None
    npu_fusion_attention_grad_v3_default = torch.ops.npu.npu_fusion_attention_grad_v3.default(primals_2, primals_1, primals_3, tangents_1, 4, 'BNSD', softmax_max = getitem_7, softmax_sum = getitem_8, softmax_in = getitem_9, attention_in = getitem_6, scale_value = 0.5, keep_prob = 0.99999999999, seed = getitem_10, offset = getitem_11);  primals_2 = primals_1 = primals_3 = tangents_1 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = None
    getitem_12 = npu_fusion_attention_grad_v3_default[0]
    getitem_13 = npu_fusion_attention_grad_v3_default[1]
    getitem_14 = npu_fusion_attention_grad_v3_default[2];  npu_fusion_attention_grad_v3_default = None
    return pytree.tree_unflatten([getitem_6, getitem_13, getitem_12, getitem_14], self._out_spec)
    