def _maybe_record_select_lane_load(
    self,
    name: str,
    result_var,
    index: sympy.Expr,
    prepared_index: sympy.Expr | None = None,
):
    """Structurally flag a stride-k select-lane load for the extract_slice
    rewrite, keyed by its result-var name.

    Detection is on the *sympy* index (not rendered text), at load() time
    where the index is still the flat pre-linearize form (e.g. ``2*x1`` for
    a ``x[...,0]`` select over a contiguous size-K inner axis). The criteria:

      * every free symbol is a free (non-reduction) range-tree node — a
        dynamic-shape stride carries a size symbol (``ks0``) that is NOT a
        range-tree node, so those bail here (correct: the lane isn't literal);
      * the index is affine with integer coefficients (no ModularIndexing /
        FloorDiv residue after removing the linear terms);
      * the minimum coefficient K is in {2,4,8} and there is NO unit-stride
        leaf — i.e. the innermost memory run is a strided gather of width K.

    The returned ``result_var`` names the exact load line codegen_body will
    rewrite, so no text classification is needed there. Gated on
    NPU_SELECT_EXTRACT_SLICE and only in linearize mode.
    """
    if not ncfg.select_extract_slice:
        return
    if not triton_codegen_linearize:
        return
    if self.inside_reduction:
        return
    var_name = getattr(result_var, "name", None)
    if not var_name or not isinstance(index, sympy.Expr):
        return

    leaves = {}
    for sym in index.free_symbols:
        node = self.range_tree_nodes.get(sym)
        if node is None or node.root.is_reduction:
            return  # non-node symbol (dynamic ks0) or reduction leaf → skip
        try:
            coeff = index.coeff(sym, 1)
        except Exception:
            return
        if not isinstance(coeff, (int, sympy.Integer)):
            return
        leaves[sym] = int(coeff)
    if not leaves:
        return
    # Affine: removing all linear terms must leave a pure constant (no sym).
    residue = index - sum(sympy.Integer(c) * s for s, c in leaves.items())
    if residue.free_symbols:
        return
    if any(c == 1 for c in leaves.values()):
        return  # a unit-stride leaf means the inner run is already contiguous
    k = min(leaves.values())
    if k not in (2, 4, 8):
        return
    if not isinstance(prepared_index, sympy.Expr):
        prepared_index = index
    # 仅对可证明的零offset静态存储范围拓宽；未知布局保留原load。
    storage_size = None
    try:
        layout = V.graph.get_buffer(name).get_layout()
        extent = layout.storage_size()
        if layout.offset == 0 and isinstance(extent, (int, sympy.Integer)) and extent > 0:
            storage_size = int(extent)
    except (AttributeError, KeyError, NotImplementedError):
        pass
    self._npu_select_lane_loads[var_name] = {
        "lane": k,
        "storage_size": storage_size,
        "index": prepared_index,
        "pointer": self.args.input(name),
    }
