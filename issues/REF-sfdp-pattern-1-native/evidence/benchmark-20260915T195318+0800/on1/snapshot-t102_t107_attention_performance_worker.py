#!/usr/bin/env python3
"""T-102～T-107 单个 SDPA pattern 的功能/性能臂。"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
SUPPORTED_PATTERNS = (*range(1, 25), 28, 29, 30)
LOW_DROPOUT_TRAINING_PATTERNS = (3, 4, 6, 7, 9, 12, 28)
REPAIRED_TRAINING_PATTERNS = (1, 2, 5)


def workload_for_pattern(pattern):
    suffix = 'half-low-dropout-training-forward' if pattern in LOW_DROPOUT_TRAINING_PATTERNS else 'half-inference'
    if pattern in REPAIRED_TRAINING_PATTERNS:
        suffix = 'half-training-forward'
    if pattern == 19:
        suffix = 'fp32-inference'
    return f'sfdp-pattern-{pattern}-registered-{suffix}'


def normalize_scale_argument(name, value):
    # c()只用于追踪占位；生产extra_check要求inv_scale为Python float/int。
    if name == 'inv_scale' and getattr(value, 'ndim', None) == 0:
        number = value.item()
        if type(number) not in (float, int):
            raise ValueError('inv_scale占位不是数值标量')
        return number
    return value


def input_description(name, value):
    if type(value) in (float, int):
        return dict(parameter=name, kind='python-scalar', python_type=type(value).__name__, value=value)
    return dict(parameter=name, shape=list(value.shape), stride=list(value.stride()),
                dtype=str(value.dtype), requires_grad=value.requires_grad)


def input_initialization_policy(pattern, index):
    # 15/20的search使用mask==0，replacement使用mask==1；必须是0/1域。
    # 18/19复用社区下三角布尔mask设计，避免随机全遮挡行改变数值合同。
    if index == 3 and pattern in (15, 18, 19, 20):
        return 'community-triangular-mask-preserve-registration-shape-dtype'
    return 'fixed-seed-normal-std0.25-or-original-scalar'


def select_registration(candidates, pattern):
    training = pattern in (*LOW_DROPOUT_TRAINING_PATTERNS, *REPAIRED_TRAINING_PATTERNS)
    suffix = '_training' if training else '_inference'
    eligible = [item for item in candidates if item[0].endswith(suffix)]
    if not eligible:
        raise ValueError(f'目标缺少所需{suffix}注册，不能用其他执行阶段替代')
    eligible.sort(key=lambda item: (('_half' in item[0]) if pattern == 19 else ('_half' not in item[0]), 'mask_fp32' in item[0],
                                    '_bs1' in item[0], item[0]))
    name, registration = eligible[0]
    if pattern == 19 and '_half' in name:
        raise ValueError('19号性能限定已验证原社区FP32域，不能自动改用已知失败half域')
    workaround = dict(registration.get('scalar_workaround', {}))
    if pattern in LOW_DROPOUT_TRAINING_PATTERNS:
        if 'dropout_p' not in workaround:
            raise ValueError('低dropout合同必须保留原dropout参数')
        # 复用社区test_fused_attention的极低非零dropout设计；不能置0后测邻接编号。
        workaround['dropout_p'] = 1e-11
    return name, registration, workaround, training


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def task_for_pattern(pattern: int) -> str:
    if pattern <= 5:
        return "T-102"
    if pattern <= 10:
        return "T-103"
    if pattern <= 15:
        return "T-104"
    if pattern <= 20:
        return "T-105"
    if pattern <= 25:
        return "T-106"
    return "T-107"


def assert_close(torch, actual, expected) -> None:
    if isinstance(actual, (tuple, list)):
        if not isinstance(expected, type(actual)) or len(actual) != len(expected):
            raise RuntimeError("输出容器结构不一致")
        for left, right in zip(actual, expected):
            assert_close(torch, left, right)
        return
    torch.testing.assert_close(actual, expected, rtol=0.2, atol=2e-3)


def read_gate(path: Path, pattern: int, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task_id": task_for_pattern(pattern),
        "acceptance_unit_id": f"AU-fuse-attention-sfdp-pattern-{pattern}",
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "numerical_execution": True,
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": workload_for_pattern(pattern),
        "worker_sha256": sha256(Path(__file__)),
        "target_control_sha256": sha256(Path(__file__).with_name('target_entry_control.py')),
        "observer_sha256": sha256(Path(__file__).with_name('native_contract_observer.py')),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能门禁字段 {key} 不符：需要 {value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("性能门禁必须记录人工复核者和时间")
    functional_records = {}
    for name in ("gpu_reference", "target_functional", "off_functional", "community_functional"):
        item = gate.get(name, {})
        source = Path(item.get("path", ""))
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file() or sha256(source) != item.get("sha256"):
            raise ValueError(f"{name}原件缺失或sha256不符")
        raw = json.loads(source.read_text())
        if name == 'gpu_reference':
            cases = [x for x in raw.get('cases',[]) if x.get('acceptance_unit_id')==expected['acceptance_unit_id']]
            if (raw.get('expected_pytorch_commit') != COMMIT or len(cases)!=1
                    or cases[0].get('tests_ran')!=1 or cases[0].get('tests_skipped')!=0
                    or cases[0].get('target_review',{}).get('expected_target')!=f'_sfdp_pattern_{pattern}'
                    or cases[0].get('target_review',{}).get('exact_target_observations',0)<1):
                raise ValueError('GPU原件没有本编号的完整原例与精确归因')
        if name in ('target_functional','off_functional'):
            functional_records[name] = raw
            mode = 'on' if name=='target_functional' else 'off'
            if (raw.get('mode')!=mode or raw.get('phase')!='functional'
                    or raw.get('correctness')!='passed' or raw.get('numerical_execution') is not True
                    or raw.get('backend')!=expected['backend']
                    or raw.get('acceptance_unit_id')!=expected['acceptance_unit_id']
                    or raw.get('pytorch_commit') != COMMIT
                    or raw.get('pytorch_worktree_status') != ''
                    or raw.get('backend_selected_before_import') is not True):
                raise ValueError(f'{name}不是对应的真实功能臂')
            for key in ('worker_sha256','target_control_sha256','observer_sha256','input_spec','registration_name','dropout_p'):
                if raw.get(key)!=gate.get(key):
                    raise ValueError(f'{name}输入/源码合同不一致: {key}')
            if ((mode=='on' and raw.get('exact_pattern_counter',0)<1)
                    or (mode=='off' and (raw.get('exact_pattern_counter')!=0
                                        or raw.get('general_fuse_attention_counter')!=0))):
                raise ValueError(f'{name}精确目标ON/OFF不成立')
        if name == 'community_functional':
            functional_records[name] = raw
            community = raw
            if (community.get('status') != 'community-contract-passed'
                    or community.get('tests_ran') != 1 or community.get('tests_skipped') != 0
                    or community.get('native_assertions_passed') is not True
                    or community.get('isolated_registration_candidate') is not False
                    or community.get('isolated_codegen_candidate', False) is not False
                    or community.get('backend') != expected['backend']
                    or community.get('acceptance_unit_id') != expected['acceptance_unit_id']
                    or community.get('pytorch_commit') != COMMIT
                    or community.get('product_gate_bypassed') is not False
                    or community.get('exact_target_observations', 0) < 1):
                raise ValueError('完整社区安装态合同未通过；不能用隔离候选签性能gate')
    validate_runtime_provenance(functional_records)
    return gate


def validate_runtime_provenance(records: dict) -> None:
    """计时在原功能环境执行：核验新进程和实际加载文件，拒绝安装态漂移。"""
    pids = [record.get('pid') for record in records.values()]
    if any(type(pid) is not int or pid < 1 for pid in pids) or len(set(pids)) != 3:
        raise ValueError('原社区、OFF、ON功能必须是三个独立进程')
    interpreters = {record.get('python_executable') for record in records.values()}
    if interpreters != {sys.executable}:
        raise ValueError('功能原件与当前计时Python环境不一致')
    checked = {}
    for name, record in records.items():
        sources = record.get('loaded_source_sha256')
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f'{name}缺少实际加载源码指纹')
        for source, digest in sources.items():
            if source in checked and checked[source] != digest:
                raise ValueError(f'功能臂之间加载源码不同：{source}')
            if source not in checked:
                file = Path(source)
                if not file.is_absolute() or not file.is_file() or sha256(file) != digest:
                    raise ValueError(f'功能验证后安装态源码变化：{source}')
                checked[source] = digest


def source_hashes() -> dict[str, str]:
    result = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file)
        if name.startswith(("torch._inductor", "torch_npu._inductor", "triton")) and path.is_file():
            result[str(path.resolve())] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=int, choices=SUPPORTED_PATTERNS, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="functional")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.pattern in (16, 17, 29):
        parser.error('该编号GPU原例命中其他目标，先修正reference映射，不能签本编号性能')
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup/runs必须为正整数")
    gate = None
    if args.phase == "benchmark":
        if args.gate is None:
            parser.error("benchmark必须提供人工签发的--gate")
        gate = read_gate(args.gate.resolve(), args.pattern, args.device)

    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        raise RuntimeError(f"必须从 {work} 启动")

    # 后端选择必须发生在 torch/torch_npu import 之前。
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_snapshots = {}
    for name in (Path(__file__).name, 'target_entry_control.py', 'native_contract_observer.py'):
        original = Path(__file__).with_name(name)
        content = original.read_bytes()
        (output / ('snapshot-' + name)).write_bytes(content)
        source_snapshots[name] = hashlib.sha256(content).hexdigest()
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(output / "debug")
    os.environ["TORCH_TRACE"] = str(output / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton-cache")

    import torch
    from torch._dynamo.utils import counters
    from torch._inductor import config
    from torch._inductor.fx_passes import fuse_attention

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载PyTorch不是冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("NPU实际注册后端不是triton_experimental")

    runtime = getattr(torch, args.device)
    runtime.set_device(0)
    device = torch.device(args.device, 0)
    torch.manual_seed(20260910 + args.pattern)
    prefix = f"_sfdp_pattern_{args.pattern}"
    candidates = []
    for name, registration in fuse_attention._get_sfdp_patterns(device):
        if name == prefix or name.startswith(prefix + "_"):
            candidates.append((name, registration))
    if not candidates:
        raise RuntimeError(f"目标{prefix}在实际设备上没有生成注册候选")
    # 上游inference注册会把dropout改为0且去重；七个训练dropout编号不能借用邻接推理图。
    registration_name, registration, scalar_workaround, training = select_registration(candidates,args.pattern)
    search_fn = registration["search_fn"]
    # 冻结社区提供的局部递归unskip包装器；不修改全局trace rules或目标guard。
    # search_fn位于torch/_inductor，直接compile会在进入FX前被MOD_SKIPLIST阻断。
    traceable_search = torch._dynamo.dont_skip_tracing(search_fn)
    model = functools.partial(traceable_search, **scalar_workaround)
    # 注册时torch.empty只用于trace，不能作为数值/性能输入。保留shape/stride/dtype，确定性初始化。
    inputs_list = []
    example_names = tuple(inspect.signature(search_fn).parameters)[:len(registration['example_inputs'])]
    if len(example_names) != len(registration['example_inputs']):
        raise ValueError('search参数与注册样例数量不一致')
    for index, value in enumerate(registration['example_inputs']):
        if value.ndim == 0:
            initialized = value.detach().clone()  # inv_scale等真实常量
        else:
            initialized = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
            if input_initialization_policy(args.pattern, index).startswith('community-triangular'):
                initialized.copy_(torch.ones(value.shape, dtype=value.dtype, device=device).tril())
            elif value.dtype == torch.bool:
                initialized.copy_(torch.rand(value.shape,device=device) > 0.5)
            else:
                initialized.normal_(mean=0.0,std=0.25)
        if training and value.requires_grad:
            initialized.requires_grad_(True)
        inputs_list.append(normalize_scale_argument(example_names[index], initialized))
    inputs = tuple(inputs_list)
    input_spec = [input_description(name, x) for name,x in zip(example_names,inputs)]
    if gate:
        for key,value in {'input_spec':input_spec,'registration_name':registration_name,
                          'dropout_p':scalar_workaround.get('dropout_p',0.0)}.items():
            if gate.get(key) != value:
                raise ValueError(f'性能输入与已签功能gate不同: {key}')
    expected = model(*inputs)

    from torch._inductor.fx_passes import joint_graph
    from torch._inductor import pattern_matcher
    from native_contract_observer import install
    install(pattern_matcher.ReplacementPatternEntry,output/'target-observations')
    joint_graph.lazy_init(device)
    target_control = {'mode':args.mode,'whole_pass_disabled':False}
    if args.mode == 'off':
        from target_entry_control import disable_entries
        target_control.update(disable_entries(joint_graph.patterns,pattern_prefix=prefix))

    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        # 两臂保留整轮joint及其他优化；OFF只移除指定编号注册。
        "use_joint_graph_passes": True,
    }
    counters.clear()
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    with config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
        actual = compiled(*inputs)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        assert_close(torch, actual, expected)
        observations = [json.loads(p.read_text()) for p in (output/'target-observations').glob('*/contract_observation.json')]
        exact_count = sum(r['target'].startswith(prefix+'_') and r['graph_changed'] for r in observations)
        general_count = counters["inductor"]["fuse_attention"]
        if args.mode == "on" and (exact_count < 1 or general_count < 1):
            raise RuntimeError(
                f"ON未命中精确pattern：exact={exact_count} general={general_count}"
            )
        if args.mode == "off" and (exact_count != 0 or general_count != 0):
            raise RuntimeError("OFF仍命中fuse_attention")

        samples = None
        memory = None
        if args.phase == "benchmark":
            for _ in range(args.warmup):
                compiled(*inputs)
            runtime.synchronize()
            runtime.reset_peak_memory_stats()
            samples = {"host_ms": [], "event_ms": []}
            for _ in range(args.runs):
                runtime.synchronize()
                start = runtime.Event(enable_timing=True)
                end = runtime.Event(enable_timing=True)
                before = time.perf_counter()
                start.record()
                compiled(*inputs)
                end.record()
                runtime.synchronize()
                samples["host_ms"].append((time.perf_counter() - before) * 1000)
                samples["event_ms"].append(start.elapsed_time(end))
            memory = {
                "allocated": runtime.max_memory_allocated(),
                "reserved": runtime.max_memory_reserved(),
            }

    torch_root = Path(torch.__file__).resolve().parents[1]
    worktree = subprocess.run(
        ["git", "-C", str(torch_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    record = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_id": task_for_pattern(args.pattern),
        "acceptance_unit_id": f"AU-fuse-attention-sfdp-pattern-{args.pattern}",
        "pattern": args.pattern,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "physical_device": os.environ.get('ASCEND_RT_VISIBLE_DEVICES' if args.device=='npu' else 'CUDA_VISIBLE_DEVICES'),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__ if args.device=='npu' else None,
        "mode": args.mode,
        "phase": args.phase,
        "backend": "triton_experimental" if args.device == "npu" else "inductor-default",
        "backend_selected_before_import": True,
        "pytorch_commit": torch.version.git_version,
        "pytorch_worktree_status": worktree,
        "correctness": "passed",
        "numerical_execution": True,
        "target_rewrite": "confirmed" if args.mode == "on" else "disabled-control",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "registration_name": registration_name,
        "exact_pattern_counter": exact_count,
        "general_fuse_attention_counter": general_count,
        "measurement_workload": workload_for_pattern(args.pattern),
        "measurement_stage": "training-forward-only" if training else "inference-forward",
        "dropout_p": scalar_workaround.get('dropout_p',0.0),
        "oracle_design": ("社区极低非零dropout方法；仍逐次比较功能输出，失败不改容差或重抽种子；计时不含backward"
            if args.pattern in LOW_DROPOUT_TRAINING_PATTERNS else
            "同输入eager对照；本次修复的training注册，计时仅前向、不含backward；原社区完整合同另绑gate"
            if training else "同输入eager对照；原社区完整合同另外绑定gate"),
        "measurement_adapter": "torch._dynamo.dont_skip_tracing(search_fn); local-wrapper-only; fullgraph-retained",
        "input_spec": input_spec,
        "off_control_scope": "remove-exact-pattern-entries-only; joint and neighboring passes stay enabled",
        "target_control": target_control,
        "input_initialization": "fixed-seed-normal-std0.25; preserve-shape-stride-dtype; inv_scale-only-trace-placeholder-to-python-scalar",
        "argument_initialization": [('inv_scale-to-python-scalar-outside-compile; preserve-value' if type(x) in (float,int)
                                     else input_initialization_policy(args.pattern, i)) for i,x in enumerate(inputs)],
        "observations": observations,
        "compile_ms": compile_ms,
        "samples": samples,
        "memory": memory,
        "timing": (
            {
                key: {"p50": percentile(value, 0.5), "p99": percentile(value, 0.99)}
                for key, value in samples.items()
            }
            if samples
            else None
        ),
        "worker_sha256": source_snapshots[Path(__file__).name],
        "target_control_sha256": source_snapshots['target_entry_control.py'],
        "observer_sha256": source_snapshots['native_contract_observer.py'],
        "source_snapshots": source_snapshots,
        "gate_sha256": sha256(args.gate) if args.gate else None,
        "loaded_source_sha256": source_hashes(),
    }
    (output / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"worker_status=passed pattern={args.pattern} mode={args.mode} "
        f"phase={args.phase} exact_pattern_counter={exact_count}"
    )


if __name__ == "__main__":
    main()
