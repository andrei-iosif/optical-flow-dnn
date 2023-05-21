import os
import matplotlib.pyplot as plt
import numpy as np


def sparsification_plot(metric_vals, metric_vals_oracle, output_path=None, label=""):
    idx = np.arange(0, 1, 0.1)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)

    ax.plot(idx, metric_vals, label=f"Ours - {label}", color='blue', linewidth=2)
    ax.plot(idx, metric_vals_oracle, label=f"Oracle - {label}", color='red', linestyle='dashed', linewidth=2)

    ax.set_xlabel('Fraction of removed pixels')
    ax.set_ylabel('Average EPE (normalized)')
    ax.set_title("Sparsification plot")

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.legend()

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, 'sparsification.jpeg'), bbox_inches="tight", pad_inches=0.3, dpi=100)
        plt.close()
    else:
        plt.show()


def sparsification_error_plot(errors_list, output_path=None):
    idx = np.arange(0, 1, 0.1)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)

    for label, error in errors_list:
        ax.plot(idx, error, label=f"{label}", linewidth=2)

    ax.set_xlabel('Fraction of removed pixels')
    ax.set_ylabel('Sparsification error')
    ax.set_title("Sparsification error plot")

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.legend()

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, 'sparsification_error.jpeg'), bbox_inches="tight", pad_inches=0.3, dpi=100)
        plt.close()
    else:
        plt.show()
