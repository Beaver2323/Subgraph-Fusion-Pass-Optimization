


def forward(self, arg0_1):
    permute = torch.ops.aten.permute.default(arg0_1, [1, 0, 2, 4, 3]);  arg0_1 = None
    unbind = torch.ops.aten.unbind.int(permute);  permute = None
    getitem = unbind[0]
    getitem_1 = unbind[1]
    getitem_2 = unbind[2];  unbind = None
    clone_default = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format);  getitem = None
    clone_default_1 = torch.ops.aten.clone.default(getitem_1, memory_format = torch.contiguous_format);  getitem_1 = None
    clone_default_2 = torch.ops.aten.clone.default(getitem_2, memory_format = torch.contiguous_format);  getitem_2 = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(clone_default, clone_default_1, clone_default_2, 4, 'BNSD', None, None, None, 0.2);  clone_default = clone_default_1 = clone_default_2 = None
    getitem_3 = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    return (getitem_3,)
    