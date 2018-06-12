import utils
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cPickle as pkl

parser = utils.parser
parser.add_argument('--data_type', default='test', help="train, test, or val")
args = parser.parse_args()

args.data_dir = '/Users/brohr/Documents/Stanford/Research/scripts/ML/ChemConvData/Datasets/data_storage/dataset'
# args.data_dir = '/Users/brohr/Documents/Stanford/Research/scripts/ML/ChemConvData/Datasets/data_storage/tiny_dataset'
# args.model_dir = '/Users/brohr/Documents/Stanford/Research/scripts/ML/ChemConvData/Experiments/brohr/50_filters_3_layers'
args.model_dir = '/Users/brohr/Documents/Stanford/Research/scripts/ML/ChemConvData/RealExperiments/brohr/50_filters_3_layers'


check_path = os.path.join(args.model_dir, 'error_analysis', args.data_type, 'labels.npy')
print check_path
assert os.path.exists(check_path)

targets = np.load(os.path.join(args.model_dir, 'error_analysis', args.data_type, 'labels.npy'))
predictions = np.load(os.path.join(args.model_dir, 'error_analysis', args.data_type, 'output.npy'))
ids = np.load(os.path.join(args.model_dir, 'error_analysis', args.data_type, 'ids.npy'))
absolute_errors = np.abs(targets - predictions)

n_atoms_path = os.path.join(args.model_dir, 'error_analysis', args.data_type, 'n_atoms.npy')
if not os.path.exists(n_atoms_path):
    print 'collecting n_atoms for each unit cell'
    n_atoms = np.zeros(ids.shape)
    for i in tqdm(range(len(ids))):
        n_atoms[i] = torch.sum(torch.load(os.path.join(args.data_dir, args.data_type, ids[i] + '.torch'))[0][3])
    np.save(n_atoms_path, n_atoms)
    print 'done collecting n_atoms'
else:
    n_atoms = np.load(n_atoms_path)




error_list = zip(ids, absolute_errors, targets, predictions, n_atoms)

error_dict = {tup[0]: (tup[1], tup[2], tup[3], tup[4]) for tup in error_list}
pkl.dump(error_dict, open(os.path.join(args.model_dir, 'error_analysis', args.data_type, 'error_dict.pkl'),'wb'))

error_list.sort(key=lambda x:x[1], reverse=True)
counter=0
n_results=20
max_atoms=100
for row in error_list:
    if counter > n_results:
        break
    if row[4] > max_atoms:
        continue
    print row
    counter+=1

plt.plot(n_atoms, absolute_errors, 'o', markersize=2)
plt.xlabel('Number of Atoms in Unit Cell')
plt.ylabel('Absolute Error (eV/atom)')
plt.title(args.data_type)
plt.show()
