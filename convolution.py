#External Modules
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch
import pdb
from CNN_input import CNNInputDataset
from ast import literal_eval

def test_forward():
    model = nn.Sequential(
        ChemConv(2,14),
        nn.ReLU(inplace = True),
        CollapseAndSum(16)
    )

    # storage_directories = ['/Users/michaeljstatt/Documents/CS230_Final_Project/data/storage_directories/150868984252']
    storage_directories = ['/Users/brohr/Documents/Stanford/Research/scripts/ML/CS230_Final_Project/150868984252']
    dataset = CNNInputDataset(storage_directories)
    return model(dataset[0])

def test_backward():
    y_pred = test_forward()
    y_actual = Variable(-0.35*torch.ones((1)))
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y_actual)
    print y_pred, y_actual, loss
    loss.backward()


def test_that_works():
    inp = Variable(torch.randn((10)))
    model = nn.Linear(10,1)
    y_pred = model(inp)
    y_actual = Variable(1.3*torch.ones((1)))
    loss_fn = nn.MSELoss()
    pdb.set_trace()
    loss = loss_fn(y_pred, y_actual)
    # print y_pred, y_actual, loss
    # pdb.set_trace()
    loss.backward()



class MyReLUFunction(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class ChemConvFunction(Function):
    @staticmethod
    def forward(self, connectivity, node_feature_matrix, filters):
        """
        connectivity is a matrix describing the degree to which each pair of
        atoms is connected

        node_feature_matrix is size N x F+2 where N is the number of atoms in the cell
        and F is the number of filters in the previous conv layer. The 2 indicates that
        strength and distance have been included as features at each layer

        filters is a matrix of size L x F_prev
        where L is the "number of atoms" in the filter, and F is the number of filters
        in the previous layer.
        """
        N = len(node_feature_matrix)
        F = len(filters)
        node_connection_matrices = make_convlayer_input_matrix(connectivity,node_feature_matrix)

        output = torch.zeros((N, F+2))
        for i_node, node_connection_matrix in enumerate(node_connection_matrices):
            for i_filter, f in enumerate(filters):
                output[i_node, i_filter] = convolution_operation(node_connection_matrix, f)

        self.save_for_backward(connectivity, node_feature_matrix, filters, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        raise NotImplementedError



class ChemConv(nn.Module):
    def __init__(self, in_depth, out_depth):
        super(ChemConv, self).__init__()
        self.in_depth = in_depth
        self.out_depth = out_depth

        ######FILTER DIMENSION#########
        filter_dimension = 12
        self.filters = torch.randn(out_depth,filter_dimension,in_depth+2)*0.01

    def forward(self, input):
        (connectivity, node_feature_matrix, energy) = input
        return ChemConvFunction.apply(connectivity, node_feature_matrix, self.filters)

class CollapseAndSum(nn.Module):
    def __init__(self, in_depth):
        super(CollapseAndSum, self).__init__()
        self.in_depth = in_depth
        self.linear = nn.Linear(in_depth, 1, bias=True)

    def forward(self, input):
        N = input.shape[0]
        output = torch.zeros((N))
        for i in range(N):
            output[i] = self.linear(input[i]).data[0]

        output = torch.sum(output)
        output = Variable(output*torch.ones((1)))
        return output


def convolution_operation(node_connection_matrix,filt):
    ordered_connection_matrix = order_input(node_connection_matrix, filt)
    if ordered_connection_matrix.shape[0] < filt.shape[0]:
        filt = filt[:ordered_connection_matrix.shape[0]]
    return torch.sum(torch.mul(ordered_connection_matrix,filt))

def order_input(node_connection_matrix,filt):
    """
    node_connection_matrix ::  e x F+1 matrix, where e is the number of edges of the
                        node that we are applying filter to and F is the number
                        of filters in the previous convlayer (or 2 for init data)

    filter        :: fx x fy matrix, where fx is the arity of the filter
                       and fy is the number of edges captured by the filter

    NOTE: for speed, we could build up the convolution operation inside the
        for loop (conv += np.dot(node_connection_matrix[best_fit]

    """
    node_connection_tensor = torch.from_numpy(node_connection_matrix)
    output_dimensions = (min(node_connection_matrix.shape[0]-1,filt.shape[0]),filt.shape[1])
    output = torch.zeros(output_dimensions)
    output[0] = node_connection_tensor[0]

    if len(filt)>0:
        i = 1
        for filtrow in filt[1:]: # presuming no atoms have NO bonds
            if i<output.shape[0]:
                scores                  = torch.matmul(node_connection_tensor,filtrow.double())
                best_fit                = np.argmax(scores)
                output[i]               = node_connection_tensor[best_fit]
                filtered_numpy          = np.delete(np.array(node_connection_tensor),best_fit,0)
                node_connection_tensor  = torch.from_numpy(filtered_numpy)
            i+=1
    return output

def make_convlayer_input_matrix(connectivity,node_feature_matrix):
    """
    Takes a connectivity list and node_feature matrix to produce an input list
    (of np arrays) for the conv layer

    connectivity :: [?x3] (list of length N)
    node_feature :: NxF matrix of features
    output ::[?xF+2] (list of length N)
    """
    output = []
    for i,connections in enumerate(connectivity):
        this_node = np.append(node_feature_matrix[i],[0,0])
        newatom = [this_node]
        for to_node, strength, dist in connections:
            node_feature = node_feature_matrix[int(to_node)]
            newatom.append(np.append(node_feature,[strength,dist])) # num_features + 2
        output.append(np.array(newatom))
    return output

if __name__ == '__main__':
    test_backward()
    # test_that_works()
