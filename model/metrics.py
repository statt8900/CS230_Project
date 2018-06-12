import numpy as np

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    return np.mean(np.abs(outputs-labels)/np.abs(labels)*100)

def r2simple(outputs, labels):
    import sklearn.metrics
    score = sklearn.metrics.r2_score(labels,outputs)
    return score

def r2(outputs, labels):
    """
    Compute the Coefficient of Determination, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) R2 in [0,1]
    """
    import sklearn.metrics
    def good(x): return not np.isnan(x) and not np.isinf(x)
    def goods((a,b)): return good(a) and good(b)
    outputs_,labels_ = zip(*filter(goods,zip(outputs,labels)))
    return sklearn.metrics.r2_score(labels_,outputs_)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {

    # 'r2simple':r2simple
    'r2':r2
    # 'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

nometrics = {}
