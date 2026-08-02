import json
import os

config_file = os.environ.get("MOTION_CONFIG", "all_datasets.json")

with open(os.path.join(os.path.dirname(__file__), "configs/" + config_file), 'r') as f:
    _cfg = json.load(f)

_overrides = os.environ.get("MOTION_CONFIG_OVERRIDES")
if _overrides:
    _parsed = json.loads(_overrides)
    unknown = set(_parsed) - set(_cfg)
    if unknown:
        raise KeyError(f"MOTION_CONFIG_OVERRIDES has keys absent from {config_file}: {sorted(unknown)}")
    _cfg.update(_parsed)

globals().update(_cfg)
