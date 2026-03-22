import json
import os

config_file = "1.json"

with open(os.path.join(os.path.dirname(__file__), "configs/" + config_file), 'r') as f:
    globals().update(json.load(f))