import numpy as np

targets = np.load(params.model_dir + 'labels.npy')
predictions = np.load(params.model_dir + 'output.npy')
ids = np.load(params.model_dir + 'ids.npy')
absolute_errors = np.abs(targets - predictions)
error_list = zip(ids, absolute_errors, targets, predictions)
error_list.sort(key=lambda x:x[1], reverse=True)
print error_list[:20]
