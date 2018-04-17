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
    def __init__(self, alpha,
                 num_module,
                 embed,
                 embed_size,
                 embed_path,
                 id_path,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.alpha = alpha

        load_id_file(id_path, embed)
        embed = load_pre_trained_emb(embed_path)
        self.entity_embeds = torch.from_numpy(embed)
        # self.entity_embeds.weight.data.copy_(torch.from_numpy(embed))
        # self.entity_embeds.weight.requires_grad=False

        self.function_modules = {}
        for id in range(num_module):
            module = ModuleBlock(embed_size, embed_size)
            self.add_module(str(id), module)
            self.function_modules[id] = module
        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

    def look_up_entity_embed(self, id):
        return self.entity_embeds[id].view(1,-1)

    def update_entity_embed(self, id, new):
        self.entity_embeds[id] = new

    def forward_path(self, path):
        x = self.look_up_entity_embed(path[0])
        length = len(path)
        for i in range(1, length, 2):
            module = self.function_modules[path[i]]
            bias = self.look_up_entity_embed(path[i+1])
            x = module(x, bias)
        w = 1/(length*self.alpha)
        output = (1-w) * bias + w * x
        self.update_entity_embed(i+1, output)
        return output

    def predict(self, ids):
        output = []
        for id in ids:
            embed = self.look_up_entity_embed(id)
            output.append(self.classifier(embed))
        output = torch.cat(output, 0)
        return output

    def forward(self, batch):
        output = []
        for path in batch:
            data_output = self.forward_path(path)
            output.append(self.classifier(data_output))
        output =torch.cat(output, 0)
        return output
