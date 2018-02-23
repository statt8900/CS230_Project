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


class Average(nn.Module):
    def __init__(self):
        super(Average, self).__init__()


    def forward(self, input):
        return AverageFunction.apply(input)

class ChemResBlockBrian(nn.Module):
    def __init__(self, num_convs, depth, filter_length, activation_fn):
        super(ChemResBlockBrian, self).__init__()
        self.conv_layers = []
        self.depth = depth
        self.filter_length = filter_length
        self.activation_fn = activation_fn
        for i in range(num_convs):
            self.conv_layers.append(ChemConv(self.depth, self.depth, self.filter_length))

    def forward(self, input):
        output = input
        for conv_layer in self.conv_layers:
            output = conv_layer(output)
            output = self.activation_fn(output)
            output = conv_layer(output)
            output += input
            output = self.activation_fn(output)
        return output

class ChemResBlock(nn.Module):
    def __init__(self, num_convs, depth, filter_length, activation_fn):
        super(ChemResBlock, self).__init__()
        self.conv_layers = []
        self.depth = depth
        self.filter_length = filter_length
        self.activation_fn = activation_fn
        for i in range(num_convs):
            self.conv_layers.append(ChemConv(self.depth, self.depth, self.filter_length))

    def forward(self, input):
        output = input
        for conv_layer in self.conv_layers:
            output = conv_layer(output)
            output = self.activation_fn(output)
            output += input
        return output

class ChemConv(nn.Module):
    def __init__(self, in_depth, out_depth, filter_length):
        super(ChemConv, self).__init__()
        self.in_depth           = in_depth
        self.out_depth          = out_depth
        self.filter_length      = filter_length

        self.property_filters   = nn.Parameter(torch.Tensor(out_depth,filter_length,in_depth))
        self.property_filters.data.normal_()
        self.property_filters.data *= 0.01

        self.bond_filters       = nn.Parameter(torch.Tensor(out_depth,filter_length,3))
        self.bond_filters.data.normal_()
        self.bond_filters.data *= 0.01

    def forward(self, node_feature_matrix):
        global connectivity_tensor, bond_property_tensor
        # (node_feature_matrix, connectivity_tensor, bond_property_tensor) = input_tup
        #Get dimensions of inputs
        (n_filters, filter_length, n_input_features) = self.property_filters.size()
        n_atoms = node_feature_matrix.size()[0]

        #Repeat the filter list and bond_property_tensor for matrix multiplication/concatenation
        property_filters_repeat     = self.property_filters.expand(n_atoms,n_filters,filter_length, n_input_features)
        bond_property_tensor_repeat = bond_property_tensor.expand(n_filters,n_atoms, filter_length, 2)
        bond_property_tensor_repeat = bond_property_tensor_repeat.transpose(0,1)

        #Dot the connectivity_tensor with the node feature matrix to create the tensors
        #to convulve with each filter
        node_connection_tensor      = torch.matmul(connectivity_tensor.transpose(1,2),node_feature_matrix)

        #Convulve the tensor with the filters
        #creates
        convolved_tensor            = torch.mul(node_connection_tensor,property_filters_repeat.transpose(0,1))
        bond_score_tensor           = torch.sum(convolved_tensor, 3).unsqueeze(3)
        bond_score_tensor           = bond_score_tensor.transpose(0,1)

        #Concatenate the bond_property_tensor onto the bond_score tensor
        #and feed into fully_connected_net
        combined_tensor             = torch.cat((bond_score_tensor, bond_property_tensor_repeat),3)

        #Combined tensor is shape: n_atoms x n_filters x filter_length x 3
        convolved_bonds             = torch.mul(combined_tensor,self.bond_filters)
        new_node_feature_matrix     = torch.sum(torch.sum(convolved_bonds,3),2)

        return new_node_feature_matrix


class SetConnectivity(nn.Module):
    def __init__(self):
        super(SetConnectivity, self).__init__()

    def forward(self,input_tup):
        global connectivity_tensor, bond_property_tensor
        (node_feature_matrix, connectivity_tensor, bond_property_tensor) = input_tup
        return node_feature_matrix
