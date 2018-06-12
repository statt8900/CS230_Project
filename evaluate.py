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
import os
import cPickle as pkl
import model.loss_functions as loss_functions
import model.metrics as metrics_module


def evaluate_error_net(model, loss_fn, dataloader, metrics, params):
    model.eval()
    summ = []
    predictions = torch.FloatTensor([])
    targets = torch.FloatTensor([])
    for batch_index, (data_batch, labels_batch) in enumerate(dataloader):
        variables = []
        for input_tensor in data_batch:
            variables.append(Variable(input_tensor))
        input_var = tuple(variables)

        labels_var = Variable(labels_batch)

        batch_output = model(input_var)
        loss = loss_fn(batch_output, labels_var)
        predictions = torch.cat(predictions, batch_output.data)
        targets = torch.cat(targets, labels_var.data)

        batch_output_npy = batch_output.data.cpu().numpy()
        labels_batch_npy = labels_var.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric:metrics[metric](batch_output_npy, labels_batch_npy)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    # import matplotlib.pyplot as plt
    # plt.plot(targets.cpu().numpy(), predictions.cpu().numpy())
    # plt.show()

    return metrics_mean

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
    labels = np.array([])
    output = np.array([])
    ids = np.char.array([])
    if params.save_activations:
        activations_path = os.path.join(params.model_dir, 'activations')
        if not os.path.isdir(activations_path):
            os.mkdir(activations_path)

        activations_save_path = os.path.join(activations_path, params.data_type)
        if not os.path.isdir(activations_save_path):
            os.mkdir(activations_save_path)

        '''for saving all in one file
        total_samples = len(dataloader.dataset)
        activations_50 = torch.zeros((total_samples, 100, params.num_filters))
        activations_30 = torch.zeros((total_samples, 100, 30))
        activations_1 = torch.zeros((total_samples, 100, 1))
        '''
    # compute metrics over the dataset
    for batch_index, (data_batch, labels_batch) in enumerate(dataloader):
        # convert to torch Variables
        (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_ids) = data_batch
        labels_batch_var                = Variable(labels_batch)
        node_property_tensor_var        = Variable(node_property_tensor)
        connectivity_tensor_var         = Variable(connectivity_tensor)
        bond_property_tensor_var        = Variable(bond_property_tensor)
        mask_atom_tensor_var            = Variable(mask_atom_tensor)

        current_batch_size = len(node_property_tensor_var)

        if params.cuda:
            node_property_tensor_var    = node_property_tensor_var.cuda(async=True)
            connectivity_tensor_var     = connectivity_tensor_var.cuda(async=True)
            bond_property_tensor_var    = bond_property_tensor_var.cuda(async=True)
            mask_atom_tensor_var        = mask_atom_tensor_var.cuda(async=True)
            labels_batch_var            = labels_batch_var.cuda(async=True)

        input_tup = (node_property_tensor_var, connectivity_tensor_var, bond_property_tensor_var, mask_atom_tensor_var, input_ids)

        # compute model output and loss
        if not params.save_activations:
            output_batch = model(input_tup)
        else:
            output_batch, activations_batch = model(input_tup)
        loss = loss_fn(output_batch, labels_batch_var)

        if params.save_activations:
            activations_batch['original_node_properties'] = node_property_tensor_var
            for key, activations_var in activations_batch.items():
                activations_tensor = activations_var.data

                save_path = os.path.join(activations_save_path, key)
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)

                activations = torch.mul(activations_tensor, mask_atom_tensor_var.data.unsqueeze(2).expand_as(activations_tensor))

                for i, input_id in enumerate(input_ids):
                    save_file = os.path.join(save_path, input_id + '.torch')
                    # save_tuple = ((activations[i].cpu(), connectivity_tensor[i], bond_property_tensor[i], mask_atom_tensor[i]), (output_batch.data[i].cpu(), labels_batch[i]))
                    save_tuple = activations[i].cpu()
                    torch.save(save_tuple, save_file)

            for input_id in input_ids:
                save_50_path = os.path.join(activations_save_path, '50')

            '''for saving all in one file
            activations_50[batch_index:batch_index+current_batch_size,:,:] = torch.mul(activations_batch['d_num_filters'].data, mask_atom_tensor.unsqueeze(2).expand_as(activations_batch['d_num_filters'].data))
            activations_30[batch_index:batch_index+current_batch_size,:,:] = torch.mul(activations_batch['d30'].data, mask_atom_tensor.unsqueeze(2).expand_as(activations_batch['d30'].data))
            activations_1[batch_index:batch_index+current_batch_size,:,:] = torch.mul(activations_batch['d1'].data, mask_atom_tensor.unsqueeze(2).expand_as(activations_batch['d1'].data))
            '''

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch_var.data.cpu().numpy()

        output = np.append(output, output_batch)
        labels = np.append(labels, labels_batch)
        for input_id in input_ids:
            ids = np.append(ids, input_id)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    if params.error_analysis:
        error_analysis_path = os.path.join(params.model_dir, 'error_analysis')
        if not os.path.isdir(error_analysis_path):
            os.mkdir(error_analysis_path)

        save_path = os.path.join(error_analysis_path, params.data_type)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    if params.error_analysis:
        np.save(os.path.join(save_path, 'ids'), ids)
        np.save(os.path.join(save_path, 'labels'), labels)
        np.save(os.path.join(save_path, 'output'), output)
        save_dict = {tup[0]: (tup[1], tup[2]) for tup in zip(ids, labels, output)}
        pkl.dump(save_dict, open(os.path.join(save_path,'dict.pkl'), 'wb'))

    import matplotlib.pyplot as plt
    plt.plot(labels, output, 'o')
    plt.show()

    '''for saving all in one file
    if params.save_activations:
        torch.save(activations_50, os.path.join(save_path, 'activations_50.torch'))
        torch.save(activations_30, os.path.join(save_path, 'activations_30.torch'))
        torch.save(activations_1, os.path.join(save_path, 'activations_1.torch'))
    '''

    return metrics_mean


if __name__ == '__main__':
    """ Evaluate the model on the test set """
    # Load the parameters
    parser = utils.parser
    parser.add_argument('--data_type', default='test', help="dataset to evaluate (test, val, train)")
    parser.add_argument('--batch_size', default=None, help="batch_size for evaluation will default to params.json value")
    parser.add_argument('--error_analysis', default=True, help="Set to 'True' to save a list of largest outliers")
    parser.add_argument('--save_activations', default=False, help="Set to 'True' to save the activations")
    args   = parser.parse_args()

    params = utils.Params(join(args.model_dir, 'params.json'))
    params = utils.get_defaults(params, args)

    # use GPU if available
    # params.cuda = torch.cuda.is_available()     # use GPU is available
    # params.data_type = args.data_type
    # params.data_dir = args.data_dir
    # params.model_dir = args.model_dir
    # params.error_analysis = args.error_analysis
    # params.save_activations = args.save_activations

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
    dataloaders = data_loader.fetch_dataloader([params.data_type], params)
    test_dl = dataloaders[args.data_type]

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = getattr(loss_functions, params.loss_fn_name)
    metrics = getattr(metrics_module, params.metrics_name)

    logging.info("Starting evaluation on the "+args.data_type+" dataset")


    # Reload weights from the saved file
    utils.load_checkpoint(join(params.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = join(args.model_dir, "metrics_{}_{}.json".format(args.data_type,args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
