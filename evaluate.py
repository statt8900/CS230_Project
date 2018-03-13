"""Evaluates the model"""

# External Modules
from os.path import isfile,join

# Internal Modules

import numpy as np
import torch
from torch.autograd import Variable
#import utils
import model.net as net
import model.data_loader as data_loader


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # summary for current eval loop
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl    = dataloaders['train']
    val_dl      = dataloaders['val']

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # convert to torch Variables
        (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor) = data_batch
        labels_batch_var                = Variable(labels_batch)
        node_property_tensor_var        = Variable(node_property_tensor)
        connectivity_tensor_var         = Variable(connectivity_tensor)
        bond_property_tensor_var        = Variable(bond_property_tensor)
        mask_atom_tensor_var            = Variable(mask_atom_tensor)

        if params.cuda:
            node_property_tensor_var    = node_property_tensor_var.cuda(async=True)
            connectivity_tensor_var     = connectivity_tensor_var.cuda(async=True)
            bond_property_tensor_var    = bond_property_tensor_var.cuda(async=True)
            mask_atom_tensor_var        = mask_atom_tensor_var.cuda(async=True)
            labels_batch_var            = labels_batch_var.cuda(async=True)

        input_tup = (node_property_tensor_var, connectivity_tensor_var, bond_property_tensor_var, mask_atom_tensor_var)

        # compute model output and loss
        output_batch = model(input_tup)
        loss = loss_fn(output_batch, labels_batch_var)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch_var.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """ Evaluate the model on the test set """
    # Load the parameters
    args   = utils.parser.parse_args()
    params = utils.Params(join(args.model_dir, 'params.json'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
