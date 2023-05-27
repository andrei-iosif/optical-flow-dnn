import numpy as np

from core.evaluation.uncertainty.sparsification_metrics import compute_ause_metric
from core.evaluation.uncertainty.sparsification_plots import sparsification_error_plot
from core.evaluation.metrics import Metrics


def create_sparsification_error_plot(results_list, labels, output_path):
    errors_list = []
    for path, label in zip(results_list, labels):
        results = Metrics()
        results.load_pickle(path)
        epe_vals_mean = results.reduce_mean("epe_vals")
        epe_vals_oracle_mean = results.reduce_mean("epe_vals_oracle")

        ause = compute_ause_metric(epe_vals_mean, epe_vals_oracle_mean)
        print(f"AUSE({label})={ause}")
        label += f" (AUSE={ause:.3f})"

        errors_list.append((label, (np.abs(epe_vals_mean - epe_vals_oracle_mean))))

    sparsification_error_plot(errors_list, output_path)
    

if __name__ == "__main__":
    results_1 = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel/raft_uncertainty_v2/results.pkl'
    results_2 = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel/ensemble_3/results.pkl'
    results_3 = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel/flow_iterations/results.pkl'
    results_4 = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel/mc_dropout_3/results.pkl'
    results_list = [results_1, results_2, results_3, results_4]
    labels = ["RAFT-Uncertainty-V2", "Ensemble-3", "FlowIterations", "Dropout-3"]
    output_path = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation/Sintel'

    create_sparsification_error_plot(results_list, labels, output_path)
