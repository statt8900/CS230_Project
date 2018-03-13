"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""
# External Modules
import json, os,sys, tabulate
import numpy as np
# Internal Modules
import utils
import model.net as net###############################################################################

def aggregate_metrics(parent_dir, metrics={}):
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

    return metrics



def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    print 'headers',headers
    print metrics
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate.tabulate(table, headers, tablefmt='pipe')

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

    m = aggregate_metrics(sys.argv[1])
    tab = metrics_to_table(m)
    with open(os.path.join(sys.argv[1],'summary.txt'), 'w') as f: f.write(tab)
