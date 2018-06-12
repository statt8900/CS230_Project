import model.datasets as datasets
import utils
import os
from model.data_loader import fetch_dataloader
import torch
from tqdm import tqdm
import numpy as np

# parser = utils.parser
# parser.add_argument('--model_dir', default=None, help="Directory containing params.json")
# args = utils.parser.parse_args()
# params = utils.Params(os.path.join(args.model_dir, 'params.json'))
#
# [dataloader] = fetch_dataloader(['train'], params)


def compile_activations_fast(path):
    ## Get filenames
    all_filenames = os.listdir(path)
    filenames = [os.path.join(path, f) for f in all_filenames if f.endswith('.torch')]

    ## check type
    nfiles = len(filenames)
    firsttensor = torch.load(filenames[0])
    assert type(firsttensor) in [torch.FloatTensor, torch.cuda.FloatTensor]
    (nrows, ncols) = firsttensor.shape

    ## instantiate output tensor with first row
    all_activations = np.zeros((nfiles*nrows, ncols))

    ## fill output tensor with remaining rows
    for i, filename in tqdm(enumerate(filenames)):
        tensor = torch.load(filename).numpy()
        all_activations[i*nrows:i*nrows+nrows] = tensor

    np.save('all_activations', all_activations)

    all_indices = range(nfiles*nrows)
    nonzeros = np.nonzero(all_activations)[0]
    nonzero_rows = np.unique(nonzeros)
    # zero_indices = [index for index in all_indices if index not in nonzero_rows]

    # return np.delete(all_activations, zero_indices, axis=0)
    return np.take(all_activations, nonzero_rows, axis=0)

def compile_activations(path):
    ## Get filenames
    all_filenames = os.listdir(path)
    filenames = [os.path.join(path, f) for f in all_filenames if f.endswith('.torch')]

    ## check type
    firsttensor = torch.load(filenames[0])
    assert type(firsttensor) in [torch.FloatTensor, torch.cuda.FloatTensor]
    (nrows, ncols) = firsttensor.shape

    ## instantiate output tensor with first row
    all_activations = firsttensor[0][:].unsqueeze(0)

    ## fill output tensor with remaining rows
    for filename in tqdm(filenames[1:]):
        tensor = torch.load(filename)
        for i in range(nrows):
            row = tensor[i][:]
            ## assuming rows of all zeros are masked
            if any(row != torch.zeros(ncols)):
                all_activations = torch.cat((all_activations, row.unsqueeze(0)), 0)

    return all_activations


if __name__ == '__main__':
    parser = utils.parser
    parser.add_argument('--save_dir', default=None, help="Directory where the answer should be saved")
    parser.add_argument('--overwrite', default=False, help="Set to True to overwrite")
    args = utils.parser.parse_args()

    save_path = os.path.join(args.save_dir, 'dataset_statistics.torch')

    if (not args.overwrite) and os.path.exists(save_path):
        raise Exception(save_path + ' already exists and overwrite is set to False.')

    path = args.data_dir

    all_activations = compile_activations_fast(path)
    all_activations = torch.from_numpy(all_activations)
    means = torch.mean(all_activations, 0)
    stdevs = torch.std(all_activations, 0)
    stats = torch.cat((means.unsqueeze(0), stdevs.unsqueeze(0)), 0)
    torch.save(stats, save_path)
