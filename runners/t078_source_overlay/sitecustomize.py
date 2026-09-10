"""未重装 wheel 时，从冻结工作树加载 T-078～T-080 涉及的产品模块。"""

import importlib.abc
import importlib.util
from pathlib import Path
import sys


SOURCE = Path(
    "/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/triton_experimental"
)
MODULES = {
    f"torch_npu._inductor.triton_experimental.{name}": SOURCE / f"{name}.py"
    for name in (
        "config",
        "overrides",
        "fx_passes",
        "lowering",
        "npu_triton_helpers",
        "npu_triton_heuristics",
        "runtime_estimation",
    )
}
MODULES[
    "torch_npu._inductor.triton_experimental.codegen.triton"
] = SOURCE / "codegen/triton.py"


class _FormalSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = MODULES.get(fullname)
        if source is None:
            return None
        if not source.is_file():
            raise ImportError(f"formal source module missing: {source}")
        return importlib.util.spec_from_file_location(fullname, source)


sys.meta_path.insert(0, _FormalSourceFinder())
sys._t078_formal_source_overlay = {  # type: ignore[attr-defined]
    name: str(path.resolve()) for name, path in MODULES.items()
}
