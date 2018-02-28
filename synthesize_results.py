"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
import numpy as np

from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')
parser.add_argument('--model_dir', default='experiments/base_model',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

def plot_loss(model_dir):
    import matplotlib.pyplot as plt

    train_loss_pth = os.path.join(model_dir,'train_loss.npy')
    val_loss_pth = os.path.join(model_dir,'val_loss.npy')

    total_train_loss = np.load(train_loss_pth)
    total_val_loss = np.load(val_loss_pth)

    fig, ax = plt.subplots()
    ax.plot(total_train_loss,color = 'r', label = 'Training MSE')
    ax.plot(total_val_loss, color = 'b', label = 'Validation MSE')
    plt.xlabel('Iterations',fontsize = 14)
    plt.ylabel('Formation Energy MSE (eV/atom)',fontsize = 14)
    plt.legend()
    plt.savefig(os.path.join(model_dir,'loss_plot.png'))
    plt.show()



if __name__ == "__main__":
    args = parser.parse_args()

    plot_loss(args.model_dir)
    # # Aggregate metrics from args.parent_dir directory
    # metrics = dict()
    # aggregate_metrics(args.parent_dir, metrics)
    # table = metrics_to_table(metrics)
    #
    # # Display the table to terminal
    # print(table)
    #
    # # Save results in parent_dir/results.md
    # save_file = os.path.join(args.parent_dir, "results.md")
    # with open(save_file, 'w') as f:
    #     f.write(table)
