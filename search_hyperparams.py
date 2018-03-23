# External Modules
import os,sys,datetime,random
from subprocess import check_call
# Internal Modules
import utils

################################################################################
"""
Peform hyperparemeters search
"""
################################################################################
PYTHON = sys.executable


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)

def vary_learning_rate(args):
    """
    Perform hypersearch over one parameter
    """
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    learning_rates = [10 ** x for x in [-5,-4]]
    for val in learning_rates:
        params.learning_rate = val
        job_name = "learning_rate_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

def vary_num_layers(args):
    """
    Perform hypersearch over one parameter
    """
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    val = [5,7,9]
    for val in vals:
        params.num_layers = val
        job_name = "num_layers_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

def vary_num_filters(args):
    """
    Perform hypersearch over one parameter
    """
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    vals = [100]
    for val in vals:
        params.num_filters = val
        job_name = "num_filters_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

if __name__=='__main__':
    parser = utils.parser
    parser.add_argument('--hyperparameter',help='Variable being optimized')
    parser.add_argument('--parent_dir', default='experiments/base_model', help="Directory containing params.json")
    args   = parser.parse_args()
    if   'learn' in args.hyperparameter: vary_learning_rate(args)
    elif 'filters' in args.hyperparameter: vary_num_filters(args)
    elif 'layers' in args.hyperparameter: vary_num_layers(args)
