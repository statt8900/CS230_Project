# External Modules
import torch,logging,json
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from os.path import join

# Internal Modules
import utils
# import model.net as net
import model.net as models
import model.loss_functions as loss_functions
import model.metrics as metrics_module
import model.data_loader as data_loader
from evaluate import evaluate, evaluate_error_net

##############################################################################
"""
Train the model
"""
##############################################################################

def train_error_net(model, optimizer, loss_fn, data_loader, metrics, params):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    loss_array = []
    # Use tqdm for progress bar
    with tqdm(total=len(data_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            # move to GPU if available

            # convert to torch Variables
            variables = []
            for input_tensor in train_batch:
                variables.append(Variable(input_tensor))
            input_var = tuple(variables)

            labels_var = Variable(labels_batch)


            # compute model output and loss
            batch_output = model(input_var)
            loss = loss_fn(batch_output, labels_var)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            ### Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                ### extract data from torch Variable, move to cpu, convert to numpy arrays
                # output_batch = batch_output.data.cpu().numpy()
                # absolute_errors = torch.abs(labels_var[0].data - labels_var[1].data)
                # labels_batch = absolute_errors.cpu().numpy()
                ### extract data from torch Variable, move to cpu, convert to numpy arrays
                batch_output_npy = batch_output.data.cpu().numpy()
                labels_batch_npy = labels_var.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](batch_output_npy, labels_batch_npy)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])
            loss_array.append(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return loss_array

def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    loss_array = []
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available

            # convert to torch Variables
            (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_ids) = train_batch
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
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch_var.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])
            loss_array.append(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return loss_array

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

        #restore all state parameters
        best_val_loss = json.load(open(join(args.model_dir, 'metrics_val_best_weights.json')))['loss']
        total_train_loss    = np.load(join(args.model_dir, 'train_loss.npy'))
        total_val_loss      = np.load(join(args.model_dir, 'val_loss.npy'))
    else:
        best_val_loss       = 1e10
        total_train_loss    = np.array([])
        total_val_loss      = np.array([])

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        if params.train_error_net:
            loss_array      =  train_error_net(model, optimizer, loss_fn, train_dataloader, metrics, params)
        else:
            loss_array      =  train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        total_train_loss    = np.append(total_train_loss,loss_array)

        # Evaluate for one epoch on validation set
        if params.train_error_net:
            val_metrics = evaluate_error_net(model, loss_fn, val_dataloader, metrics, params)
        else:
            val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_loss = val_metrics['loss']
        is_best  = val_loss <= best_val_loss
        total_val_loss = np.append(total_val_loss, [val_loss]*len(loss_array))

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new lowest loss")
            best_val_loss = val_loss

            # Save best val metrics in a json file in the model directory
            best_json_path = join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        np.save(join(model_dir,"train_loss"),total_train_loss)
        np.save(join(model_dir,"val_loss"),total_val_loss)
####################################################################################

if __name__ == '__main__':

    # Load the parameters from json file
    parser = utils.parser
    parser.add_argument('--data_type', default='train', help="dataset to evaluate (test, val, train)")
    parser.add_argument('--save_activations', default=False, help="Set to 'True' to save the activations")
    parser.add_argument('--error_analysis', default=True, help="Set to 'True' to save a list of largest outliers")
    parser.add_argument('--train_error_net', default=False, help="Set to 'True' to use the train_error_net function")
    args = utils.parser.parse_args()


    params = utils.Params(join(args.model_dir, 'params.json'))
    params = utils.get_defaults(params, args)
    # params.save_activations = args.save_activations
    # params.error_analysis = args.error_analysis
    # params.data_type = args.data_type
    # if args.model_dir:
    #     params.model_dir = args.model_dir
    # if args.data_dir:
    #     params.data_dir = args.data_dir
    # if 'model_name' not in params.dict.keys():
    #     params.model_name = 'Net'
    # if 'loss_fn_name' not in params.dict.keys():
    #     params.loss_fn_name = 'MSELoss'
    # if 'metrics_name' not in params.dict.keys():
    #     params.metrics_name = 'metrics'

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    # if params.train_error_net == True:
    #     dataloaders = data_loader.fetch_dataloader(['val', 'test'], params)
    #     train_dl    = dataloaders['val']
    #     val_dl      = dataloaders['test']
    # else:
    #     dataloaders = data_loader.fetch_dataloader(['train', 'val'], params)
    #     train_dl    = dataloaders['train']
    #     val_dl      = dataloaders['val']

    dataloaders = data_loader.fetch_dataloader(['val', 'test'], params)
    train_dl    = dataloaders['val']
    val_dl      = dataloaders['test']


    logging.info("- done.")

    # Define the model and optimizer
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    model_class = getattr(models, params.model_name)
    model = model_class(params)
    if params.cuda:
        model.cuda()
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    params.weight_decay = 0 if 'weight_decay' not in params.dict.keys() else params.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay = params.weight_decay)

    # fetch loss function and metrics
    loss_fn = getattr(loss_functions, params.loss_fn_name)
    # loss_fn_instance = loss_fn(params)
    loss_fn_instance = loss_fn()
    metrics = getattr(metrics_module, params.metrics_name)


    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn_instance, metrics, params, args.model_dir,
                   args.restore_file)
