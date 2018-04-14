import numpy as np
import math
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModuleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModuleBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, bias):
        return F.relu(F.linear(input, self.weight, bias), inplace=True)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)


class ModuleNet(nn.Module):
    def __init__(self, num_entity,
                 num_module,
                 num_bias,
                 embed_size,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.entity_embeds = nn.Embedding(num_entity, embed_size)
        self.bias_embeds = nn.Embedding(num_bias, embed_size)
        self.function_modules = {}
        for id in range(num_module):
            module = ModuleBlock(embed_size, embed_size)
            self.add_module(str(id), module)
            self.function_modules[id] = module
        self.classifier = nn.Sequential(nn.Linear(embed_size, 1, bias=True),
                                        nn.Sigmoid())

    def look_up_entity_embed(self, id):
        lookup_tensor = torch.LongTensor([id]).cuda()
        return self.entity_embeds(autograd.Variable(lookup_tensor))

    def look_up_bias_embed(self, id):
        lookup_tensor = torch.LongTensor([id]).cuda()
        return self.bias_embeds(autograd.Variable(lookup_tensor))

    def forward_path(self, path):
        x = self.look_up_entity_embed(path[0])
        length = len(path[:-2])
        for i in range(1, length, 2):
            module = self.function_modules[path[i]]
            bias = self.look_up_bias_embed(path[i+1])
            x = module(x, bias)
        module = self.function_modules[path[-2]]
        bias = self.look_up_entity_embed(path[-1])
        output = module(x, bias) - bias
        return output

    def forward(self, batch):
        output = []
        for paths in batch:
            data_output = (self.forward_path(paths[0]) + self.forward_path(paths[1])) / 2
            output.append(self.classifier(data_output))
        output =torch.cat(output, 0)
        return output
