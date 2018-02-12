
# coding: utf-8

# In[ ]:

#External Modules
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch
import pdb
from CNN_input import CNNInputDataset
from ast import literal_eval
import matplotlib.pyplot as plt
from torch.autograd import gradcheck

import cPickle as pkl
from copy import deepcopy


# In[396]:


def test_forward():
    model = nn.Sequential(
        ChemConv(2,14),
        nn.ReLU(inplace = True),
        nn.Linear(16,1, bias=True),
        Average()
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

def store_params(model):
    p = []
    for i in model.parameters():
        p.append(i)

    return p



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



# In[476]:

class ChemConvFunction(Function):
    @staticmethod
    def forward(ctx, node_feature_matrix, filters, connectivity):
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

        #### Grad Check Hack
        # connectivity = pkl.load(open('connectivity_hack.pkl', 'r'))
        ####

        n_atoms = len(node_feature_matrix) # number of atoms
        [n_filters, filter_length, n_descriptors_in] = filters.shape
        node_connection_matrices, atom_index_vectors = make_convlayer_input_matrix(connectivity,node_feature_matrix)
        output = torch.zeros((n_atoms, n_filters))
        ordered_connection_matrices_save = torch.zeros((n_atoms, n_filters, filter_length, n_descriptors_in)) # 3 14 13 4
        # atom_index_vectors_save = torch.zeros((n_filters, filter_length)).type(torch.LongTensor)
        one_hots = torch.zeros((n_atoms, n_filters, n_atoms, filter_length)) # there are n_atoms*n_filters of these one_hot matricies. Each is size n_atoms by filter_length because one_hot*filter must be size n_atoms * n_descriptors_in

        for i_node, (node_connection_matrix, atom_index_vector) in enumerate(zip(node_connection_matrices, atom_index_vectors)):
            for i_filter, f in enumerate(filters):
                conv_result, ordered_connection_matrix, atom_index_vector, filter_used = convolution_operation(node_connection_matrix, atom_index_vector, f)
                # print i_node, i_filter
                output[i_node, i_filter] = conv_result
                ordered_connection_matrices_save[i_node, i_filter, :, :] = ordered_connection_matrix
                ### Start here - there's an atom index vector of length 11 messing everything up. Perhaps get one hot encoding here.
                one_hot = get_one_hot(atom_index_vector, n_atoms, filter_length)
                one_hots[i_node, i_filter, :, :] = one_hot
                # torch_atom_index_vector = torch.from_numpy(np.array(atom_index_vector))
                # atom_index_vectors_save[i_filter, :] = torch_atom_index_vector

        ctx.filters = filters
        ctx.ordered_connection_matrices_save = ordered_connection_matrices_save
        ctx.one_hots = one_hots

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (n_atoms, n_filters, filter_length, n_descriptors_in) = ctx.ordered_connection_matrices_save.shape
        filters = ctx.filters
        ordered_connection_matrices_save = ctx.ordered_connection_matrices_save
        one_hots = ctx.one_hots

        grad_input = Variable(torch.zeros((n_atoms, n_descriptors_in)))
        grad_filters = Variable(torch.zeros((n_filters, filter_length, n_descriptors_in)))
        for i_node in range(n_atoms):
            for i_filter in range(n_filters):
                grad_filters.data += grad_output[i_node, i_filter].data * ordered_connection_matrices_save[i_node, i_filter, :, :]
                grad_input.data += grad_output[i_node, i_filter].data * torch.matmul(one_hots[i_node, i_filter, :, :], filters[i_filter, :, :])


#         raise NotImplementedError
#         N = len(ctx.atom_index_vectors)
#         D = ctx.filters_used[0].shape[1]
#         print ctx.filters_used.shape
#         grad_input = torch.zeros((N, D))
# #         grad_filters = torch.zeros()
#         for (node_connection_matrix, atom_index_vector, filter_used) in zip(ctx.ordered_connection_matrices, ctx.atom_index_vectors, ctx.filters_used):
#             for filt_row in filter_used:
#                 pass


        grad_connectivity = None
        # print grad_input
        return grad_input, grad_filters, grad_connectivity


def convolution_operation(node_connection_matrix, atom_index_vector, filt):
    ordered_connection_matrix, atom_index_vector = order_input(node_connection_matrix, atom_index_vector, filt)
    if ordered_connection_matrix.shape[0] < filt.shape[0]:
        new_connection_matrix = torch.zeros(filt.shape)
        L = ordered_connection_matrix.shape[0]
        new_connection_matrix[:L, :] = ordered_connection_matrix
        ordered_connection_matrix = new_connection_matrix
    output = torch.sum(torch.mul(ordered_connection_matrix,filt))
    return output, ordered_connection_matrix, atom_index_vector, filt

def get_one_hot(atom_index_vector, N, filter_length):
    # N is number of atoms in this case
    D = len(atom_index_vector)
    output = torch.zeros(N,filter_length)
    for (location, atom_index) in enumerate(atom_index_vector):
        output[atom_index][location] += 1
    return output


class ChemConv(nn.Module):
    def __init__(self, in_depth, out_depth):
        super(ChemConv, self).__init__()
        self.in_depth = in_depth
        self.out_depth = out_depth

        ######FILTER DIMENSION#########
        filter_dimension = 13
        self.filters = nn.Parameter(torch.Tensor(out_depth,filter_dimension,in_depth+2))
        self.filters.data.normal_()
        self.filters.data *= 0.01

    def forward(self, input_tup):
        (node_feature_matrix_var, connectivity) = input_tup
        return ChemConvFunction.apply(node_feature_matrix_var, self.filters, connectivity)


def order_input(node_connection_matrix, atom_index_vector, filt):
    """
    node_connection_matrix ::  e x F+1 matrix, where e is the number of edges of the
                        node that we are applying filter to and F is the number
                        of filters in the previous convlayer (or 2 for init data)

    filter        :: fx x fy matrix, where fx is the arity of the filter
                       and fy is the number of edges captured by the filter

    NOTE: for speed, we could build up the convolution operation inside the
        for loop (conv += np.dot(node_connection_matrix[best_fit]

    """
    sum_check = sum(atom_index_vector) # for assert check at end of function
    node_connection_tensor = torch.from_numpy(node_connection_matrix)
    output_dimensions = (min(node_connection_matrix.shape[0],filt.shape[0]),filt.shape[1]) #smaller of (num edges this node has, length of filter)
    output = torch.zeros(output_dimensions)
    output[0] = node_connection_tensor[0]
    filtered_numpy = np.delete(np.array(node_connection_tensor),0,0) # Brian

    ordered_atom_index_vector = [] # Brian
    if len(filt)>0:
        i = 1
        ordered_atom_index_vector.append(atom_index_vector[0]) # Brian
        del atom_index_vector[0] # Brian
        for filtrow in filt[1:]: # presuming no atoms have NO bonds
            if i<output.shape[0]:
                node_connection_tensor  = torch.from_numpy(filtered_numpy) # Brian
                scores                  = torch.matmul(node_connection_tensor,filtrow.double())
                best_fit                = np.argmax(scores)
                ordered_atom_index_vector.append(atom_index_vector[best_fit]) # Brian
                del atom_index_vector[best_fit] # Brian
                output[i]               = node_connection_tensor[best_fit]
                filtered_numpy          = np.delete(np.array(node_connection_tensor),best_fit,0)
            i+=1

    assert sum_check == sum(ordered_atom_index_vector) # these vectors should contain the same indices in a possibly different order, so they should have the same sum
    return output, ordered_atom_index_vector

def make_convlayer_input_matrix(connectivity,node_feature_matrix):
    """
    Takes a connectivity list and node_feature matrix to produce an input list
    (of np arrays) for the conv layer

    connectivity :: [?x4] (list of length N)
    node_feature :: NxF matrix of features
    output ::[?xF+2] (list of length N)
    """
    output = []
    atom_index_vectors = []
    for i,connections in enumerate(connectivity):
        this_node = np.append(node_feature_matrix[i],[0,0])
        newatom = [this_node]
        atom_index_vector = [i]
        for to_node, strength, dist in connections:
            node_feature = node_feature_matrix[int(to_node)]
            atom_index_vector.append(int(to_node))
            newatom.append(np.append(node_feature,[strength,dist])) # num_features + 2
        output.append(np.array(newatom))
        atom_index_vectors.append(atom_index_vector)
        assert len(atom_index_vector) == len(newatom)
    assert len(atom_index_vectors) == len(output)
    return output, atom_index_vectors






# In[477]:

##### ChemConv Test
mode = 'chem'

if mode == 'chem':
    model = nn.Sequential(
        ChemConv(2,5),
        nn.ReLU(inplace = True),
        nn.Linear(5,1, bias=True),
        Average()
    )

    # storage_directories = ['/Users/michaeljstatt/Documents/CS230_Final_Project/data/storage_directories/150868984252']
    storage_directories = ['/Users/brohr/Documents/Stanford/Research/scripts/ML/CS230_Final_Project/150868984252']
    dataset = CNNInputDataset(storage_directories)


    ### See if forward pass causes an error
    tup = dataset[0]
    (connectivity, node_feature_matrix, energy) = tup
    node_feature_matrix_var = Variable(torch.from_numpy(node_feature_matrix))
    input_tup = (node_feature_matrix_var, connectivity)

    y_pred = model(input_tup)



    #### See if backward pass causes an error
    y_actual = Variable(-0.35*torch.ones((1)))
    loss_fn = nn.MSELoss()

    loss = loss_fn(y_pred, y_actual)


    # eps = 0.001
    # node_feature_matrix_eps = deepcopy(node_feature_matrix)
    # node_feature_matrix_eps[0][1] += eps
    # node_feature_matrix_eps_var = Variable(torch.from_numpy(node_feature_matrix_eps))
    # input_tup_eps = (node_feature_matrix_eps_var, connectivity)
    # y_pred_eps = model(input_tup_eps)
    # loss_eps = loss_fn(y_pred_eps, y_actual)
    # print (loss_eps.data[0] - loss.data[0])/eps

    p=1
    params = store_params(model)
    loss.backward()


    ### Gradient checking
    '''
    connectivity_hack = pkl.dump(connectivity, open('connectivity_hack.pkl', 'w'))
    input = (node_feature_matrix_var, Variable(torch.randn(5,13,4), requires_grad=True), Variable(torch.zeros(1), requires_grad = False))
    # test = gradcheck(ccf.apply, input, eps=1e-6, atol=1e-4)
    test = gradcheck(ChemConvFunction.apply, input, eps=1e-6, atol=1e-4)
    print test
    '''




    ### try to overfit one data point

    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9)

    num_epochs = 500
    loss_history = []
    for i in range(num_epochs):
        y_pred = model(input_tup)
        loss = loss_fn(y_pred, y_actual)
        loss_history.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print loss_history[-1]
    plt.plot(range(num_epochs), loss_history)
    plt.xlabel('Number of times the model has seen the same dang structure')
    plt.ylabel('Mean square error')
    plt.show()





# In[480]:

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
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

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
        return MyLinearFunction.apply(input, self.weight, self.bias)


if mode == 'linear':
    inp = Variable(torch.randn((1,10)))
    model = MyLinearModule(10,1)
    y_pred = model(inp)
    y_actual = Variable(1.3*torch.ones((1)))
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y_actual)
    # print y_pred, y_actual, loss
    # pdb.set_trace()
    loss.backward()

    # f = y_pred.grad_fn
    # print f
    # f.apply(2)

    p = store_params(model)


# In[479]:

#### Average Test






# In[478]:

'''
inp = Variable(torch.randn((10,5))) #10 atoms (N), 5 descriptors (D)


# model = nn.Sequential(
#     nn.Linear(5,1, bias=True),
#     Average()
# )

model = nn.Sequential(
    MyLinearModule(5,1, bias=True),
    Average()
)


# model = nn.Sequential(
#     MyLinearModule(5,1, bias=True)
# )


y_pred = model(inp)
# y_pred.backward()
y_actual = Variable(1.3*torch.ones((1)))
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_actual)
loss.backward()
'''

# In[ ]:
