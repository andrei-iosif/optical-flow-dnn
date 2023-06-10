import numpy as np


class Metrics:
    def __init__(self) -> None:
        self.vals = {}

    def add(self, metric_name, metric_val):
        if metric_name not in self.vals.keys():
            self.vals[metric_name] = []
        
        self.vals[metric_name].append(metric_val)

    def add_dict(self, metrics_dict):
        for metric_name in metrics_dict.keys():
            self.add(metric_name, metrics_dict[metric_name])

    def reduce_mean(self, compute_std=False):
        result = {}

        for metric_name in self.vals.keys():
            if metric_name.startswith("out_") and isinstance(self.vals[metric_name][0], np.ndarray):
                metric_vals = np.concatenate(self.vals[metric_name])
            else:
                metric_vals = np.array(self.vals[metric_name])

            mean_val = np.mean(metric_vals)
            result[metric_name] = mean_val

            if compute_std:
                std_val = np.std(metric_vals)
                result[f"{metric_name}_std"] = std_val
        return result
