"""在未重装 wheel 时，仅从 T-078 冻结工作树加载五个 Python 产品模块。"""

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
    )
}


class _FormalSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = MODULES.get(fullname)
        if source is None:
            return None
        if not source.is_file():
            raise ImportError(f"T-078 formal source module missing: {source}")
        return importlib.util.spec_from_file_location(fullname, source)


sys.meta_path.insert(0, _FormalSourceFinder())
sys._t078_formal_source_overlay = {  # type: ignore[attr-defined]
    name: str(path.resolve()) for name, path in MODULES.items()
}
