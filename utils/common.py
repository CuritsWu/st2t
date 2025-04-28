from copy import deepcopy


def deep_update(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for k, v in patch.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged
