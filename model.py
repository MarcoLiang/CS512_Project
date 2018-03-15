import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LENGTH = 2
NUM_METAPATH = 100
NUM_AUTHOR = 100
EMBED_SIZE = 2
learning_rate = 0.01


####### to do ########
a = np.array([[1., 2.], [2., 3.], [3., 4.]])
author_embedding = Variable(torch.from_numpy(a).float(), requires_grad=False)
metapath_weight = Variable(torch.Tensor(2, 2), requires_grad=True)
metapath_bias = Variable(torch.Tensor(2, 2), requires_grad=True)
fc1_weight = Variable(torch.Tensor(LENGTH, EMBED_SIZE*2), requires_grad=True)
fc2_weight = Variable(torch.Tensor(LENGTH, 1), requires_grad=True)



path = [0,0,1,1,2]
length = 2
data = [[[0,0,1],[1,0,2],[1,1,2]],[[0,0,1,1,2],[1,0,0,0,2]]]


def compose_path(path, length):
    current_state = author_embedding[path[0]]
    for i in range(length):
        next_state = author_embedding[path[2*(i+1)]]
        metapath = path[2*i+1]
        w = metapath_weight[metapath]
        b = metapath_bias[metapath]
        current_state = current_state * w * next_state + b
    output1 = current_state

    current_state = author_embedding[path[-1]]
    for i in reversed(range(length)):
        metapath = path[2*i+1]
        next_state = author_embedding[path[2*i]]
        w = metapath_weight[metapath]
        b = metapath_bias[metapath]
        current_state = current_state * w * next_state + b

    output2 = current_state
    return torch.cat([output1,output2])


def calculate_group(paths, length):
    output = 0
    for path in paths:
        output += compose_path(path, length)
    return output / len(paths)


def combine_goups(data):
    length = 0
    for group in data:
        length += 1
        if length == 1:
            output = F.relu(torch.matmul(calculate_group(group, length), fc1_weight[length-1]))
            continue
        z = F.relu(torch.matmul(calculate_group(group, length), fc1_weight[length-1]))
        output = torch.cat([output, z])
    return output


combined_output = combine_goups(data)
output = F.sigmoid(torch.matmul(combined_output, fc2_weight))


output.backward()

metapath_weight.data.sub_(metapath_weight.grad.data * learning_rate)
metapath_bias.data.sub_(metapath_bias.grad.data * learning_rate)
fc1_weight.data.sub_(fc1_weight.grad.data * learning_rate)
fc2_weight.data.sub_(fc2_weight.grad.data * learning_rate)


