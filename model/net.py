"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Internal Modules
from CS230_Project.model import modules as custom_nn

#####################
#!!!IMPORTANT!!!
#--------------------
#Cannot change filter_length as filter_length is set in the data extraction step
#in build_dataset.py


#Dataset Filter Length
###############################
#UPDATE WHEN DATASET IS UPDATED
filter_length = 13
###############################


class Net(nn.Module):
    """
    Neural Net
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains model parameters
        """
        super(Net, self).__init__()

        #Set Global variables
        self.setglobals     = custom_nn.SetGlobalVars()

        #Convolutional Layers
        self.chemconv1      = custom_nn.ChemConv(7,params.num_filters,filter_length)
        self.chemresblock1  = custom_nn.ChemResBlock(3,params.num_filters,filter_length,nn.ReLU(inplace = True))

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1            = nn.Linear(params.num_filters,30, bias=True)
        self.relu1          = nn.ReLU(inplace = True)
        self.fc2            = nn.Linear(30,1, bias=True)

        #Average layer to determine average formation_energy_per_atom
        self.average        = custom_nn.Average()


    def forward(self, batch_input):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of input tuples.
                Each tuple contains

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        out     =   self.setglobals(batch_input)
        out     =   self.chemconv1(out)
        out     =   self.chemresblock1(out)
        out     =   self.fc1(out)
        out     =   self.relu1(out)
        out     =   self.fc2(out)
        out     =   self.average(out)

        return out

loss_fn = torch.nn.MSELoss()

# def loss_fn(outputs, labels):
#     """
#     Compute the cross entropy loss given outputs and labels.
#
#     Args:
#         outputs: (Variable) dimension batch_size x 6 - output of the model
#         labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
#
#     Returns:
#         loss (Variable): cross entropy loss for all images in the batch
#
#     Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
#           demonstrates how you can easily define a custom loss function.
#     """
#     num_examples = outputs.size()[0]
#     return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    return np.mean(np.abs(outputs-labels)/np.abs(labels)*100)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    # 'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
