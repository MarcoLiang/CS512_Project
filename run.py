import argparse
parser = argparse.ArgumentParser()
import numpy as np
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data import Data
from model import ModuleNet

# module options
max_length = 3
num_metapath = 3
num_entity = 3
embed_size = 128
classifier_first_dim = 128
classifier_second_dim = 32

# Optimization options
learning_rate = 0.01
batsize = 2
num_iterations = 100
check_every = 1000

data_set = Data()
data_dir = "./_reduced_dataset/filter_venue_since_2005/pattern"
data_set.data_load(data_dir, max_length)
for X_batch, y_batch in data_set.next_batch(data_set.X_train, data_set.y_train):
    print(X_batch)
    print(y_batch)


# toy data
# there are 2 data in 1 batch
# label = [1, 0]
# paths = [[[[0,0,1,2],[0,1,1,3]],[[0,0,2,1,1,3],[0,0,2,0,1,4]]],
#          [[[1,1,2,1],[1,0,2,2],[1,2,2,3]],[[1,0,0,1,2,3],[1,0,0,0,2,5]]]]
#
# dataset = [[paths, label]] # only one batch


def train_model(dataset, embed_size, num_entity, num_metapath, max_length, classifier_first_dim, classifier_second_dim,
                num_iterations, check_every):
    execution_engine = ModuleNet(embed_size=embed_size,
                                 num_entity=num_entity,
                                 num_metapath=num_metapath,
                                 max_length=max_length,
                                 classifier_first_dim=classifier_first_dim,
                                 classifier_second_dim=classifier_second_dim).cuda()
    execution_engine.train()
    optimizer = torch.optim.Adam(execution_engine.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()

    t = 0
    epoch = 0
    while t < num_iterations:
        epoch += 1
        print('Starting epoch %d' % epoch)
        for batch in dataset:
            t += 1
            paths, labels = batch
            labels_var = Variable(torch.FloatTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, labels_var.view(-1, 1))
            loss.backward()
            optimizer.step()
            print(t, loss.data[0])
            if t % check_every == 0:
                check_accuracy()


def check_accuracy():
    pass


train_model(data_set, embed_size, num_entity, num_metapath, max_length, classifier_first_dim, classifier_second_dim,
            num_iterations, check_every)

# def main(args):
#     train_model(embed_size, num_entity, num_metapath, max_length, classifier_first_dim, classifier_second_dim, num_iterations, check_every)
#
#
# if __name__ == '__main__':
#   args = parser.parse_args()
#   main(args)
