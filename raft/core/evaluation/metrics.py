import os
import pickle
import numpy as np


class Metrics:
    def __init__(self) -> None:
        self.metrics = {}
        self.num_samples = 0

    def add(self, sample_id, metric_name, metric_value):
        if sample_id not in self.metrics.keys():
            self.metrics[sample_id] = {}
            self.num_samples += 1

        self.metrics[sample_id][metric_name] = metric_value

    def reduce_mean(self, metric_name):
        sample_ids = list(self.metrics.keys())
        sample_id = sample_ids[0]
        if metric_name not in self.metrics[sample_id].keys():
            raise RuntimeError(f"Metric {metric_name} not found in results dict!")
        
        sample = self.metrics[sample_id][metric_name]
        if isinstance(sample, np.ndarray):
            result_sum = np.zeros_like(sample)
        elif isinstance(sample, np.float64):
            result_sum = 0.0
        else:
            raise NotImplementedError(f"Reduce method not implemented for argument of type: {type(sample)}")
        
        num_samples = 0
        for sample_id in self.metrics.keys():
            val = self.metrics[sample_id][metric_name]
            result_sum += val
            num_samples += 1
        
        return result_sum / num_samples

    def save_pickle(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, "results.pkl")

        with open(path, 'wb') as file:
            pickle.dump(self.metrics, file)
        print(f"Serialized results to: {path}")

    def load_pickle(self, file_path):
        with open(file_path, "rb") as file:
            self.metrics = pickle.load(file)
