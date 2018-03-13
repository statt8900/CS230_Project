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

def vary_learning_rate():
    """
    Perform hypersearch over one parameter
    """
    args   = utils.parser.parse_args()
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    learning_rates = [10 ** random.uniform(-6, 1) for _ in range(5)]
    for val in learning_rates:
        params.learning_rate = val
        job_name = "learning_rate_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

def vary_filter_depth():
    """
    Perform hypersearch over one parameter
    """
    args   = utils.parser.parse_args()
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    filter_depths = [random.randint(10,100) for _ in range(5)]
    for val in filter_depths:
        params.filter_depth = val
        job_name = "filter_depth_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

def vary_filter_width():
    """
    Perform hypersearch over one parameter
    """
    args   = utils.parser.parse_args()
    params = utils.Params(os.path.join(args.parent_dir, 'params.json'))
    filter_widths = [random.randint(10,30) for _ in range(5)]
    for val in filter_widths:
        params.filter_width = val
        job_name = "filter_width_"+str(val)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
