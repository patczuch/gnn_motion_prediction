import json
import os

config_file = "all_datasets.json"

with open(os.path.join(os.path.dirname(__file__), "configs/" + config_file), 'r') as f:
    globals().update(json.load(f))