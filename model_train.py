import matplotlib.pyplot as plt
import sys, math
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from time import time
from sklearn.model_selection import train_test_split


import CS230_Project.convolution as conv
from CS230_Project.CNN_input import CNNInputDataset
from CS230_Project.misc.utils import BatchesBar
################################################################################
"""
This module contains the PyTorchTrainer object used to train models
"""
################################################################################

class PyTorchTrainer(object):
    def __init__(self, dataset, train_loader, model, optimizer, loss_fn, val_dataloader = None):
        self.dataset        = dataset
        self.train_loader     = train_loader
        self.val_dataloader = val_dataloader
        self.num_train_batches    = len(train_loader)
        self.num_val_batches = len(val_loader)
        self.model          = model
        self.optimizer      = optimizer
        self.loss_fn        = loss_fn

        self.train_loss_history = []
        self.val_loss_history = []
        self.old_time  = None
        self.epoch     = 0

    def train_epochs(self, num_epochs):
        self.start     = time()
        self.curr_time = self.start
        for i in range(num_epochs):
            print '\n'
            print 'Starting epoch {0}/{1}'.format(self.epoch+1, num_epochs)
            self.train_epoch()
        # self.plot_loss(num_epochs)

    def train_epoch(self):
        self.epoch += 1
        losses      = []
        self.model.train()
        progressbar = BatchesBar(max = self.num_batches)
        for i, batch in enumerate(self.train_loader):

            (node_property_tensor, connectivity_tensor, bond_property_tensor, e_form) = batch
            y_actual = Variable(e_form)

            node_property_tensor_var    = Variable(node_property_tensor)
            connectivity_tensor_var     = Variable(connectivity_tensor)
            bond_property_tensor_var    = Variable(bond_property_tensor)

            input_tup = (node_property_tensor_var, connectivity_tensor_var, bond_property_tensor_var)
            y_out     = self.model(input_tup)

            loss = self.loss_fn(y_out, y_actual)

            losses.append(loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progressbar.MSE_curr = np.mean(np.array(losses))
            progressbar.next()
        self.train_loss_history.append(np.mean(np.array(losses)))

        #Print output of epoch

        self.old_time   = self.curr_time
        self.curr_time  = time()
        print ''
        print 'Total time since start: {}'.format(self.curr_time - self.start)
        print 'Epoch Time: {}'.format(self.curr_time - self.old_time)
        print 'Training MSE = {}'.format(self.train_loss_history[-1])
        print '##########################'

        self.validate_epoch()

    def validate_epoch(self):
        losses      = []
        self.model.eval()
        progressbar = BatchesBar(max = self.num_batches)
        for i, batch in enumerate(self.val_dataloader):

            (node_property_tensor, connectivity_tensor, bond_property_tensor, e_form) = batch
            y_actual = Variable(e_form)

            node_property_tensor_var    = Variable(node_property_tensor)
            connectivity_tensor_var     = Variable(connectivity_tensor)
            bond_property_tensor_var    = Variable(bond_property_tensor)

            input_tup = (node_property_tensor_var, connectivity_tensor_var, bond_property_tensor_var)
            y_out     = self.model(input_tup)

            loss = self.loss_fn(y_out, y_actual)
            losses.append(loss.data[0])

            progressbar.MSE_curr = np.mean(np.array(losses))
            progressbar.next()
        self.val_loss_history.append(np.mean(np.array(losses)))

        #Print output of epoch

        self.old_time   = self.curr_time
        self.curr_time  = time()
        print ''
        print 'Validation'
        print 'Total time since start: {}'.format(self.curr_time - self.start)
        print 'Validation Time: {}'.format(self.curr_time - self.old_time)
        print 'Validation MSE = {}'.format(self.val_loss_history[-1])
        print '##########################'

    def plot_loss(self):
        num_epochs = len(self.train_loss_history)
        plt.plot(range(num_epochs), self.train_loss_history, label='training')
        plt.plot(range(num_epochs), self.val_loss_history, label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Mean square error (eV/atom) squared')
        plt.title('Final MSE = %1.3f'%(self.val_loss_history[-1]))
        plt.legend(loc='best')
        plt.show()

# def train_test_split(inds, test_size, train_inds):
#     pass

def generate_train_val_dataloader(dataset, batch_size, split=0.9, n_samples=10):
    """
    return two Data`s split into training and validation
    `split` sets the train/val split fraction (0.9 is 90 % training data)
    u
    """
    ## this is a testing feature to make epochs go faster, uses only some of the available data
    # if n_samples < 1.:
    #     n_samples = int(n_samples * len(dataset))
    # else:
    #     n_samples = len(dataset)
    inds = np.arange(n_samples)
    train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)

    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_inds),
        batch_size=batch_size,
        # shuffle=shuffle,
        # num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_inds),
        batch_size=batch_size,
        # shuffle=shuffle,
        # num_workers=num_workers
    )
    return train_loader, val_loader


def train():
    #Model Iterations/Sampling Parameters
    num_epochs      = 10
    # limit         = 64
    n_samples       = 64
    split           = 0.7
    batch_size      = 8
    # sampler       = SubsetRandomSampler(range(limit))
    #Optimizer parameters
    learning_rate   = 1e-4
    momentum        = 0.9
    #Model paramters
    model = nn.Sequential(
        conv.SetConnectivity(),
        conv.ChemConv(7,100,13),
        conv.ChemResBlock(3,100,13,nn.ReLU(inplace = True)),
        nn.Linear(100,30, bias=True),
        nn.ReLU(inplace = True),
        nn.Linear(30,1, bias=True),
        conv.Average()
    )

    #Create necessary objects for input
    dataset    = CNNInputDataset(limit=n_samples)
    train_loader, val_loader = generate_train_val_dataloader(dataset, batch_size, split=split, n_samples=n_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = torch.nn.MSELoss()
    #Create Trainer object
    trainer = PyTorchTrainer(dataset, train_loader, model, optimizer,loss_fn, val_loader)

    #Train on the data
    trainer.train_epochs(num_epochs)
    # trainer.plot_loss()

if __name__ == '__main__':
    train()
