import os
from datetime import datetime

def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def time_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def res_file(results_path, true_model_name, prediction_model_name):
    if true_model_name:
        return f"{results_path}/{true_model_name}_{prediction_model_name}_{time_now()}.jsonl"
    else:
        return f"{results_path}/{prediction_model_name}_{time_now()}.jsonl"