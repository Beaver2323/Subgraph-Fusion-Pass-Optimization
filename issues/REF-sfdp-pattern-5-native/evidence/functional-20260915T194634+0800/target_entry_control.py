"""仅移除指定目标的注册项；不关闭整轮 pass，不扩展任何设备/product guard。"""
from __future__ import annotations


def disable_entries(registry, *, handler_names=(), pattern_prefix=None):
    removed = []
    retained = 0
    for key, entries in list(registry.patterns.items()):
        keep = []
        for entry in entries:
            handler = getattr(getattr(entry, 'handler', None), '__name__', '')
            pattern = getattr(entry, 'pattern_name', '')
            selected = handler in handler_names or (
                pattern_prefix is not None and isinstance(pattern, str)
                and (pattern == pattern_prefix or pattern.startswith(pattern_prefix + '_'))
            )
            if selected:
                removed.append(pattern or handler)
            else:
                keep.append(entry)
        registry.patterns[key] = keep
        retained += len(keep)
    if not removed:
        raise ValueError('未找到目标注册，拒绝伪造 OFF')
    return {'removed': removed, 'retained_entries': retained, 'whole_pass_disabled': False}
