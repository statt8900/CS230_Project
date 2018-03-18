"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""
# External Modules
import json, os,sys, tabulate
import numpy as np
# Internal Modules
import utils
import model.net as net###############################################################################

def aggregate_metrics(parent_dir, metrics={},data_type = 'train'):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_{}_best.json'.format(data_type))
    print data_type,metrics_file
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics, data_type= data_type)

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
    ax.set_yscale('log')
    plt.xlabel('Iterations',fontsize = 14)
    plt.ylabel('Formation Energy MSE (eV/atom)',fontsize = 14)
    plt.legend()
    plt.savefig(os.path.join(model_dir,'loss_plot.png'))
    plt.show()

def parity_plot(model_dir, data_types = ['test']):
    import matplotlib.pyplot as plt

    fig, ax         = plt.subplots(figsize=(9,8))
    font_size_curr  = 25
    label_dict      = {'test'         :'Test'
                        ,'val'        :'Validation'
                        ,'train'      :'Training'}

    R2_dict         = {'test'         :'0.938'
                            ,'val'    :'0.937'
                            ,'train'  :'0.991'}

    for data_type, color  in zip(data_types,['r','b','g']):
        test_output = np.load(os.path.join(model_dir,data_type+'_output.npy'))
        test_labels = np.load(os.path.join(model_dir,data_type+'_labels.npy'))
        ax.scatter(test_labels, test_output,color = color,s= 2, label = label_dict[data_type]+'; $\\mathrm{r}^\\mathrm{2}$ = '+R2_dict[data_type])

    plt.plot([-5,5],[-5,5],linestyle='--',color='k')
    plt.xlabel('Actual $\\mathrm{E}_\\mathrm{F}$ (eV/atom)',fontsize = font_size_curr)
    plt.ylabel('Predicted $\\mathrm{E}_\\mathrm{F}$ (eV/atom)',fontsize = font_size_curr)
    plt.title('M-50-3 after 60 epochs', fontsize = font_size_curr+3)
    ax.set_xlim([-4,1])
    ax.set_ylim([-4,1])
    # Plot legend.
    lgnd = plt.legend(loc="upper left", numpoints=1, fontsize=font_size_curr-5)
    #change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [15]
    lgnd.legendHandles[1]._sizes = [15]
    lgnd.legendHandles[2]._sizes = [15]
    plt.setp(ax.get_xticklabels(), fontsize=font_size_curr)
    plt.setp(ax.get_yticklabels(), fontsize=font_size_curr)
    plt.savefig(os.path.join(model_dir,'parity_plot.png'), dpi=1000)
    # plt.show()


if __name__ == "__main__":

    m = aggregate_metrics(sys.argv[1], data_type = sys.argv[2])
    tab = metrics_to_table(m)
    with open(os.path.join(sys.argv[1],'summary.txt'), 'w') as f: f.write(tab)
