import json
import os


def save_dict_json(d, filename="data", output_dir=""):
    output_path = os.path.join(output_dir, f"{filename}.json")
    with open(output_path, 'w') as fp:
        json.dump(d, fp, indent=0)


def load_dict_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)

    return data
