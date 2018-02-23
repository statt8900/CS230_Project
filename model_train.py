import CS230_Project.convolution as conv
from CS230_Project.CNN_input import CNNInputDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from time import time

class PyTorchTrainer(object):
    def __init__(self, dataset, dataloader, model, optimizer, loss_fn):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn


        self.epoch_loss_history = []
        self.old_time = None
        self.epoch = 0

    def train_epochs(self, num_epochs):
        self.start = time()
        self.curr_time = self.start
        for i in range(num_epochs):
            print 'Starting epoch {0}/{1}'.format(self.epoch, num_epochs)
            self.train_epoch()
        self.plot_loss(num_epochs)
    def train_epoch(self):
        self.epoch += 1
        losses = []
        for tup in self.dataset:
            (node_feature_matrix, connectivity_tensor, bond_property_tensor, e_form) = tup
            y_actual = Variable(e_form*torch.ones(1))

            node_feature_matrix_var     = Variable(node_feature_matrix)
            connectivity_tensor_var     = Variable(connectivity_tensor)
            bond_property_tensor_var    = Variable(bond_property_tensor)

            input_tup = (node_feature_matrix_var, connectivity_tensor_var, bond_property_tensor_var)
            y_out     = self.model(input_tup)

            loss = self.loss_fn(y_out, y_actual)
            losses.append(loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epoch_loss_history.append(np.mean(np.array(losses)))

        #Print output of epoch
        self.old_time = self.curr_time
        self.curr_time = time()
        print 'Total time since start: {}'.format(self.curr_time - self.start)
        print 'Epoch Time: {}'.format(self.curr_time - self.old_time)
        print 'MSE Error = {}'.format(self.epoch_loss_history[-1])
        print '##########################'
    def plot_loss(self,num_epochs):
        plt.plot(range(num_epochs), self.epoch_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Mean square error (eV/atom) squared')
        plt.title('Final MSE = {0}'.format(self.epoch_loss_history[-1]))
        plt.show()


def train():
    #Model Iterations/Sampling Parameters
    num_epochs      = 10
    limit           = 10
    batch_size      = 10
    sampler         = SubsetRandomSampler(range(limit))
    #Optimizer parameters
    learning_rate   = 1e-3
    momentum        = 0.9
    #Model paramters
    model = nn.Sequential(
        conv.SetConnectivity(),
        conv.ChemConv(2,100,13),
        conv.ChemResBlock(3,100,13,nn.ReLU(inplace = True)),
        nn.Linear(100,30, bias=True),
        nn.ReLU(inplace = True),
        nn.Linear(30,1, bias=True),
        conv.Average()
    )

    #Create necessary objects for input
    dataset     = CNNInputDataset(limit = limit)
    dataloader  = DataLoader(dataset,
                             sampler=SubsetRandomSampler(range(limit)),
                             batch_size=batch_size
                             )
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)
    loss_fn = torch.nn.MSELoss()
    #Create Trainer object

    trainer = PyTorchTrainer(dataset, dataloader, model, optimizer,loss_fn)

    #Train on the data
    trainer.train_epochs(num_epochs)

if __name__ == '__main__':
    train()
