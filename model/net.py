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
        self.save_activations = params.save_activations


        #Set Global variables
        self.setglobals     = custom_nn.SetGlobalVars()

        #Convolutional Layers
        self.chemconv1      = custom_nn.ChemConv(7,params.num_filters,filter_length)
        self.chemresblock1  = custom_nn.ChemResBlock(params.num_layers,params.num_filters,filter_length,nn.ReLU(inplace = True))

        # Mask Layer
        self.mask = custom_nn.Mask()

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1            = nn.Linear(params.num_filters,30, bias=True)
        self.relu          = nn.ReLU(inplace = True)
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
        if self.save_activations:
            activations_batch = {}

        out     =   self.setglobals(batch_input)
        out     =   self.chemconv1(out)
        out     =   self.chemresblock1(out)
        if self.save_activations:
            out = self.mask(out)
            activations_batch['d_num_filters'] = out

        out = self.relu(out)
        out     =   self.fc1(out)

        if self.save_activations:
            out = self.mask(out)
            activations_batch['d30'] = out

        out     =   self.relu(out)
        out     =   self.fc2(out)
        out = self.mask(out)

        if self.save_activations:
            activations_batch['d1'] = out

        out     =   self.average(out)

        if not self.save_activations:
            return out
        else:
            return out, activations_batch


class ErrorNet(nn.Module):
    def __init__(self, params):
        """
        Args:
            params: (Params) contains model parameters
        """
        super(ErrorNet, self).__init__()


        self.n_neurons      = params.n_neurons
        self.n_res_units    = params.n_res_units
        self.input_size     = params.input_size

        self.set_globals    = custom_nn.ErrorNetSetGlobals()
        self.fc1            = nn.Linear(self.input_size,self.n_neurons, bias=True)
        self.activation_fn  = nn.ReLU(inplace = True)
        self.resfcblock     = custom_nn.ResFCBlock(n_neurons=self.n_neurons, n_units=self.n_res_units, activation_fn = self.activation_fn, bias = True)
        self.fc_last        = nn.Linear(self.n_neurons, 1, bias=True)
        self.mask           = custom_nn.Mask()

        #Average layer to determine average formation_energy_per_atom
        self.average        = custom_nn.Average()
        self.maskandaverage = custom_nn.MaskAndAverage()
        self.maskandsum = custom_nn.MaskAndSum()

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
        out = self.set_globals(batch_input)
        out = self.fc1(out)

        out = self.activation_fn(out)
        out = self.resfcblock(out)
        out = self.fc_last(out)

        # out = self.mask(out)
        # out = self.average(out)
        # out = self.maskandaverage(out)
        out = self.maskandsum(out)

        return out

class ErrorNetConv(nn.Module):
    """
    Neural Net
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains model parameters
        """
        super(ErrorNetConv, self).__init__()
        self.save_activations = params.save_activations


        #Set Global variables
        self.setglobals     = custom_nn.SetGlobalVars()

        #Convolutional Layers
        self.chemconv1      = custom_nn.ChemConv(7,params.num_filters,filter_length)
        self.chemresblock1  = custom_nn.ChemResBlock(params.num_layers,params.num_filters,filter_length,nn.ReLU(inplace = True))

        # Mask Layer
        self.mask = custom_nn.Mask()

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1            = nn.Linear(params.num_filters,30, bias=True)
        self.relu          = nn.ReLU(inplace = True)
        self.fc2            = nn.Linear(30,1, bias=True)

        #Average layer to determine average formation_energy_per_atom
        self.average        = custom_nn.Average()
        self.maskandaverage = custom_nn.MaskAndAverage()


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
        if self.save_activations:
            activations_batch = {}

        out     =   self.setglobals(batch_input)
        out     =   self.chemconv1(out)
        out     =   self.chemresblock1(out)
        if self.save_activations:
            out = self.mask(out)
            activations_batch['d_num_filters'] = out

        out = self.relu(out)
        out     =   self.fc1(out)

        if self.save_activations:
            out = self.mask(out)
            activations_batch['d30'] = out

        out     =   self.relu(out)
        out     =   self.fc2(out)

        if self.save_activations:
            activations_batch['d1'] = out

        out     =   self.maskandaverage(out)
        out     =   torch.abs(out)

        if not self.save_activations:
            return out
        else:
            return out, activations_batch






# loss_fn = torch.nn.MSELoss()

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
