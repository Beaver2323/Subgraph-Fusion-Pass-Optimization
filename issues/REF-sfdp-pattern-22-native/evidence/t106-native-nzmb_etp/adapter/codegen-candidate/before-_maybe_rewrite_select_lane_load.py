def _maybe_rewrite_select_lane_load(self):
    """Rewrite a stride-k select-lane load into a contiguous full load +
    register-level ``extract_slice``.

    A ``x[..., 0]`` select on the innermost *input* axis of size K collapses
    into a flat address ``K*x2 + ...`` with no coeff-1 leaf. bishengir sees a
    strided, sub-16B inner run and degrades the access to an element-granular
    gather (the freqs / row-broadcast slowness). Re-adding a local lane
    arange ``[0, K)`` in the innermost *tensor* slot makes the inner run
    contiguous (one StridedSlice DMA burst); ``extract_slice`` then squeezes
    the lane back out, restoring the original tile shape so every downstream
    op is byte-for-byte unchanged.

    Detection is done structurally at load() time on the sympy index (see
    ``_maybe_record_select_lane_load``), which records each qualifying load
    by its result-var name in ``self._npu_select_lane_loads``. This method
    only has to (a) find that exact line by ``lhs == var`` — no text
    classification, no affine re-parse — and (b) emit the rewrite, reusing
    the line's own already-rendered ptr/addr/mask so the address text is
    never reconstructed. Gated on NPU_SELECT_EXTRACT_SLICE, once per kernel.

    Only fires in real-block linearize mode (needs the per-slot tile tokens
    and ``var_tensor_dims``); non-linearize kernels are left untouched.
    """
    if not ncfg.select_extract_slice:
        return
    if getattr(self, "_npu_select_slice_done", False):
        return
    if self.inside_reduction:
        return
    if not (triton_codegen_linearize and getattr(self, "_linearize_applied", False)):
        return
    targets = getattr(self, "_npu_select_lane_loads", None)
    if not targets:
        return

    # slot -> node-name across all free (non-reduction) trees.
    slot_to_name = {}
    name_to_token = {}
    for tree in self.range_trees:
        if tree.is_reduction:
            continue
        for nm, slot in (getattr(tree, "var_tensor_dims", {}) or {}).items():
            slot_to_name[slot] = nm
        name_to_token.update(
            getattr(tree, "node_block_constexpr", {}) or {}
        )
    if not slot_to_name:
        return
    ndim = self.triton_tensor_ndim()
    inner_slot = ndim - 1
    inner_name = slot_to_name.get(inner_slot)
    if inner_name is None:
        return

    # Build extract sizes once (real per-slot tile tokens; inner slot -> 1).
    sizes = []
    for s in range(ndim):
        if s == inner_slot:
            sizes.append("1")
            continue
        tok = name_to_token.get(slot_to_name.get(s))
        if tok is None:
            return  # a needed tile token is missing → leave loads untouched
        sizes.append(tok)
    offsets = ", ".join(["0"] * ndim)
    strides = ", ".join(["1"] * ndim)
    bcast = self.indexing_size_str(inner_slot)

    new_lines = []
    counter = 0
    changed = False
    for line in self.body._lines:
        if not isinstance(line, str):
            new_lines.append(line)
            continue
        parsed = _npu_parse_tl_load_assignment(line)
        if parsed is None or parsed[1] not in targets:
            new_lines.append(line)
            continue
        indent, lhs, value_ast, load_ast = parsed
        target = targets[lhs]
        lane = target["lane"]
        prepared_index = target["index"]
        direct_inner, indirect_inner = self._select_index_node_relation(
            prepared_index, inner_name
        )
        #   * Case B (strided SLICE, x[:, ::k]): inner axis kept, iterated at stride k,
        #   * Case A (select, x[..., 0]): inner axis absent/broadcast, squeeze to size 1.
        if direct_inner:
            # Case B (strided slice) reads K× the inner bytes, discards K-1/K — a net
            # loss for a bandwidth-bound plain slice (0.98× vs strided gather). Unlike
            # Case A, the dropped lanes are REAL distinct elements, no free burst to win.
            # Gated by select_extract_slice_strided (default ON): target is a strided
            # sub-16B run feeding a broadcast store (freqs); disable when slice BW wins.
            if not ncfg.select_extract_slice_strided:
                new_lines.append(line)
                continue
            rewritten = self._emit_strided_slice_extract(
                indent, lhs, value_ast, load_ast, target, inner_name,
                inner_slot, ndim, slot_to_name, name_to_token, bcast,
                offsets, counter,
            )
            if rewritten is None:
                # Precondition unmet (not a clean k*inner term, missing tile
                # token, or no inner mask to bound the widened tail) → leave
                # the strided load untouched rather than risk truncation.
                new_lines.append(line)
            else:
                new_lines.extend(rewritten)
                counter += 1
                changed = True
            continue
        if indirect_inner:
            new_lines.append(line)
            continue

        lane_var = f"_es_lane{counter}"
        full_var = f"_es_full{counter}"
        counter += 1
        new_lines.append(f"{indent}{lane_var} = tl.arange(0, {lane}){bcast}")
        original_pointer = ast.unparse(load_ast.args[0])
        load_ast.args[0] = ast.parse(
            f"({original_pointer}) + {lane_var}", mode="eval"
        ).body
        ast.fix_missing_locations(value_ast)
        new_lines.append(f"{indent}{full_var} = {ast.unparse(value_ast)}")
        # Emit the slice via the registered first-class op (single source of
        # truth for the extract_slice text; see NPUTritonKernelOverrides).
        slice_expr = self.overrides.extract_slice(
            full_var,
            "[" + offsets + "]",
            "[" + ", ".join(sizes) + "]",
            "[" + strides + "]",
        )
        new_lines.append(f"{indent}{lhs} = {slice_expr}")
        changed = True

    if changed:
        self.body._lines = new_lines
    self._npu_select_slice_done = True
