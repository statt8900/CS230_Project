#External Modules
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch
import pdb
from CNN_input import CNNInputDataset
from ast import literal_eval

global connectivity_tensor, bond_property_tensor


################################################################################
"""
This module contains classes to describe the layers of our NN
"""
################################################################################


###########################
#ChemResBlock NN module
#--------------------------

class ChemResBlock(nn.Module):
    """
    A Residual Block that loops through a set of ChemConv layers.
    After every two layers the input to that set of two layers is added the output
    of the final layer before the activation function is applied.

    Notes:
    Forces the input and output of each ChemConv layer in the block to have
    the same in_depth and out_depth
    """
    def __init__(self, num_convs, depth, filter_length, activation_fn):
        """
        input:
        num_convs       :: int
                        number of ChemConv layers in the block
        depth           :: int
                        in_depth and out_depth of each ChemConv layer
        filter_length   :: int
                        Length of each of the filters (i.e. number of bonds compared)
        activation_fn   :: PyTorch Activation Module
        """
        super(ChemResBlock, self).__init__()
        self.conv_layers   = []
        self.depth         = depth
        self.filter_length = filter_length
        self.activation_fn = activation_fn

        for i in range(num_convs):
            self.conv_layers.append(ChemConv(self.depth, self.depth, self.filter_length))

    def forward(self, input):
        #Iterate through each of the ChemConv layers
        output = input
        for conv_layer in self.conv_layers:
            #Apply first of two ChemConv layers
            output = conv_layer(output)
            #Feed output into activiation function
            output = self.activation_fn(output)
            #Apply second of two ChemConv layers
            output = conv_layer(output)
            #Add input to this iteration to the output
            output += input
            #Apply second activation function
            output = self.activation_fn(output)
        return output

###########################
#ChemConv NN module
#--------------------------


class ChemConv(nn.Module):
    """
    The chemical convolution operation
    Convolves a node_property_tensor with a set of filters to determine the local
    chemical enviroment of each node
    !!!Requires the SetConnectivity module to be used prior to this module!!!
    """
    def __init__(self, in_depth, out_depth, filter_length):
        """
        input:
        in_depth        :: int
                        number of properties on input node_property_tensor
        out_depth       :: int
                        number of properties on output node_property_tensor
        filter_length   :: int
                        Length of each of the filters (i.e. number of bonds compared)
        """
        super(ChemConv, self).__init__()
        self.in_depth           = in_depth
        self.out_depth          = out_depth
        self.filter_length      = filter_length

        self.filters   = nn.Parameter(torch.Tensor(out_depth,filter_length,in_depth+2))
        # nn.init.xavier_normal(self.filters)
        # self.filters.data[:in_depth+1]  *= 0.01
        # #
        self.filters   = nn.Parameter(torch.Tensor(out_depth,filter_length,in_depth+2))
        self.filters.data.normal_()
        self.filters.data *= 1/self.in_depth

    def forward(self, node_property_tensor):
        """
        forward pass for the ChemConv layer
        !!!Requires the SetConnectivity module to be used prior to this module!!!
        input:
        node_property_tensor        :: PyTorch tensor size (A atoms x self.in_depth Properties)
        output:
        new_node_property_tensor    :: PyTorch tensor size (A atoms x self.out_depth Propterties)
        """

        #Gets the global variables for this samples connectivity
        global connectivity_tensor, bond_property_tensor
        import pdb; pdb.set_trace()
        #Get dimensions of inputs
        (n_filters, filter_length, n_input_features) = self.filters.size()
        n_atoms = node_property_tensor.size()[0]

        #Repeat the filter list and bond_property_tensor for matrix multiplication/concatenation
        filters_repeat                  = self.filters.expand(n_atoms,*self.filters.size())

        #Dot the connectivity_tensor with the node feature matrix to create the tensors
        #to convolve with each filter
        node_connection_tensor          = torch.matmul(connectivity_tensor.transpose(1,2),node_property_tensor)

        #Convulve the tensor with the filters
        combined_tensor                 = torch.cat((node_connection_tensor, bond_property_tensor),2)
        convolved_tensor                = torch.mul(combined_tensor,filters_repeat.transpose(0,1))

        #Combined tensor is shape: n_atoms x n_filters x filter_length x 3
        new_node_property_tensor        = torch.sum(torch.sum(convolved_tensor,3),2)
        new_node_property_tensor        = new_node_property_tensor.transpose(0,1)

        return new_node_property_tensor

###########################
#SetConnectivity NN module
#--------------------------

class SetConnectivity(nn.Module):
    """
    First step of the model sets the global variables containing the connectivity
    information of the input graph. These global variables are used in every
    ChemConv layer
    """
    def __init__(self):
        super(SetConnectivity, self).__init__()

    def forward(self,input_tup):
        #Takes in the input from the dataset object
        #Sets the two global variables connectivity_tensor and bond_property_tensor
        #passes the node
        global connectivity_tensor, bond_property_tensor
        (node_property_tensor, connectivity_tensor, bond_property_tensor) = input_tup
        return node_property_tensor


###########################
#Average NN module
#--------------------------

class Average(nn.Module):
    """
    Module that takes in an input vector and outputs the average of that vector
    """
    def __init__(self):
        super(Average, self).__init__()

    def forward(self, input):
        return AverageFunction.apply(input)

class AverageFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.N = input.shape[0]
        output = torch.mean(input)
        output = output*torch.ones(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = Variable(grad_output.data/float(ctx.N)*torch.ones(ctx.N,1))
        return grad_input

###########################
#Archived Code
#--------------------------
# class ChemResBlock(nn.Module):
#     def __init__(self, num_convs, depth, filter_length, activation_fn):
#         super(ChemResBlock, self).__init__()
#         self.conv_layers = []
#         self.depth = depth
#         self.filter_length = filter_length
#         self.activation_fn = activation_fn
#         for i in range(num_convs):
#             self.conv_layers.append(ChemConv(self.depth, self.depth, self.filter_length))
#
#     def forward(self, input):
#         output = input
#         for conv_layer in self.conv_layers:
#             output = conv_layer(output)
#             output = self.activation_fn(output)
#             output += input
#         return output
