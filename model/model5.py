import numpy as np
import math
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.load_embedding import *


class ModuleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModuleBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, bias=None, flag=True):
        if flag:
            return F.relu(F.linear(input, self.weight, bias), inplace=True)
        else:
            return F.relu(F.linear(input, self.weight, self.bias), inplace=True)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)


class ModuleNet(nn.Module):
    def __init__(self, alpha,
                 num_module,
                 embedding,
                 embed_size,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.entity_embeds = embedding
        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

        self.function_modules = {}
        for id in range(num_module):
            module = ModuleBlock(embed_size, embed_size)
            self.add_module(str(id), module)
            self.function_modules[id] = module

    def look_up_embed(self, id):
        lookup_tensor = torch.LongTensor([id]).cuda()
        return self.entity_embeds(autograd.Variable(lookup_tensor))

    def look_up_embeds(self, ids):
        lookup_tensor = torch.LongTensor(ids).cuda()
        return self.entity_embeds(autograd.Variable(lookup_tensor))

    def update_embed(self, id, new):
        self.entity_embeds.weight.data[id] = new.data

    def forward_path(self, path):
        x = self.look_up_embed(path[0])
        length = len(path)
        for i in range(1, length-2, 2):
            module = self.function_modules[path[i]]
            # x = module(x)
            bias = self.look_up_embed(path[i+1])
            x = module(x, bias)
        module = self.function_modules[path[i+2]]
        x = module(x, flag=False)
        # x = F.normalize(x)
        # w = 1/(length*self.alpha)
        # w=0.5
        # output = (1-w) * bias + w * x
        # self.update_embed(i+1, output)
        return x

    def predict(self, ids):
        embeds = self.look_up_embeds(ids)
        output = self.classifier(embeds)
        return output

    def forward(self, batch):
        output = []
        for path in batch:
            output.append(self.forward_path(path))
        output = torch.cat(output, 0)
        output = self.classifier(output)
        return output
