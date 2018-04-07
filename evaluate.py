"""Evaluates the model"""

# External Modules
from os.path import isfile,join

# Internal Modules

import numpy as np
import torch,logging
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader


def evaluate(model, loss_fn, dataloader, metrics, params, error_analysis = False):
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
    test_labels = np.array([])
    test_output = np.array([])
    test_ids = np.char.array([])
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # convert to torch Variables
        (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_ids) = data_batch
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

        input_tup = (node_property_tensor_var, connectivity_tensor_var, bond_property_tensor_var, mask_atom_tensor_var, input_ids)

        # compute model output and loss
        output_batch = model(input_tup)
        loss = loss_fn(output_batch, labels_batch_var)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch_var.data.cpu().numpy()

        test_output = np.append(test_output, output_batch)
        test_labels = np.append(test_labels, labels_batch)
        for input_id in input_ids:
            test_ids = np.append(test_ids, input_id)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    if error_analysis:
        np.save(params.model_dir + 'test_labels', test_labels)
        np.save(params.model_dir + 'test_output', test_output)
        np.save(params.model_dir + 'test_ids', test_ids)

    return metrics_mean


if __name__ == '__main__':
    """ Evaluate the model on the test set """
    # Load the parameters
    parser = utils.parser
    parser.add_argument('--data_type', default='test', help="dataset to evaluate (test, val, train)")
    parser.add_argument('--batch_size', default=None, help="batch_size for evaluation will default to params.json value")
    parser.add_argument('--error_analysis', default=True, help="Set to 'True' to save a list of largest outliers")
    args   = parser.parse_args()
    params = utils.Params(join(args.model_dir, 'params.json'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available
    params.data_dir = args.data_dir
    params.model_dir = args.model_dir

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the "+args.data_type+ " dataset...")

    # fetch dataloaders
    if not args.batch_size is None:
        params.batch_size = args.batch_size
    dataloaders = data_loader.fetch_dataloader([args.data_type], args.data_dir, params)
    test_dl = dataloaders[args.data_type]

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation on the "+args.data_type+" dataset")

    # Reload weights from the saved file
    utils.load_checkpoint(join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, error_analysis=args.error_analysis)
    save_path = join(args.model_dir, "metrics_{}_{}.json".format(args.data_type,args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    if args.error_analysis:
        targets = np.load(params.model_dir + 'test_labels.npy')
        predictions = np.load(params.model_dir + 'test_output.npy')
        ids = np.load(params.model_dir + 'test_ids.npy')
        absolute_errors = np.abs(targets - predictions)
        error_list = zip(ids, absolute_errors, targets, predictions)
        error_list.sort(key=lambda x:x[1], reverse=True)
        print error_list[:20]
