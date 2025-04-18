import json


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data
