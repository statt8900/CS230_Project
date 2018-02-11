
# coding: utf-8

# In[113]:

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

#         self.save_for_backward(connectivity, node_feature_matrix, filters, output)
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
        self.filters = nn.Parameter(torch.Tensor(out_depth,filter_dimension,in_depth+2))
        self.filters.data.normal_()
        self.filters.data *= 0.01

    def forward(self, input):
        (connectivity, node_feature_matrix, energy) = input
        return ChemConvFunction.apply(connectivity, node_feature_matrix, self.filters)

class CollapseAndSumFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        raise NotImplementedError

    @staticmethod
    def backward():
        raise NotImplementedError



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
        output.requires_grad=True
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

def store_params(model):
    p = []
    for i in model.parameters():
        p.append(i)

    return p


# In[148]:

class MyLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        print grad_output
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        print ctx.needs_input_grad
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        print grad_input
        return grad_input, grad_weight, grad_bias

class MyLinearModule(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MyLinearModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        print type(input.data)
        print type(self.weight.data)
        print type(self.bias.data)
        return MyLinearFunction.apply(input, self.weight, self.bias)



# In[149]:

#### my_linear_test():
'''
inp = Variable(torch.randn((1,10)))
model = MyLinearModule(10,1)
y_pred = model(inp)
y_actual = Variable(1.3*torch.ones((1)))
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_actual)
print y_pred, y_actual, loss
# pdb.set_trace()
loss.backward()

# f = y_pred.grad_fn
# print f
# f.apply(2)

p = store_params(model)
# return p

# p = my_linear_test()
'''

# In[72]:

##### ChemConv Test
'''
model = nn.Sequential(
    ChemConv(2,14),
    nn.ReLU(inplace = True),
    CollapseAndSum(16)
)

# storage_directories = ['/Users/michaeljstatt/Documents/CS230_Final_Project/data/storage_directories/150868984252']
storage_directories = ['/Users/brohr/Documents/Stanford/Research/scripts/ML/CS230_Final_Project/150868984252']
dataset = CNNInputDataset(storage_directories)
y_pred = model(dataset[0])
# pdb.set_trace()
y_actual = Variable(-0.35*torch.ones((1)))
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_actual)
# print y_pred, y_actual, loss
p=1
p = store_params(model)
loss.backward()
'''




# In[169]:

class CollapseAndAverageFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = torch.matmul(input, weight)
        output += bias
        output = torch.mean(output)
        output = Variable(output*torch.ones(1))
        output.requires_grad=True
        print output
        pdb.set_trace()
        return output

    @staticmethod
    def backward():
        raise NotImplementedError



class CollapseAndAverage(nn.Module):
    def __init__(self, in_depth):
        super(CollapseAndAverage, self).__init__()
        self.in_depth = in_depth
        self.weight = nn.Parameter(torch.Tensor(in_depth, 1))
        self.bias = nn.Parameter(torch.Tensor(1))


    def forward(self, input):
        CollapseAndAverageFunction.apply(input, self.weight, self.bias)


# In[170]:

#### CollapseAndAverage Test
inp = Variable(torch.randn((10,5))) #10 atoms (N), 5 descriptors (D)
model = CollapseAndAverage(5)
y_pred = model(inp)
asd
y_actual = Variable(1.3*torch.ones((1)))
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_actual)
print y_pred, y_actual, loss
# pdb.set_trace()
loss.backward()

# f = y_pred.grad_fn
# print f
# f.apply(2)

p = store_params(model)
# return p


# In[ ]:




# In[ ]:
