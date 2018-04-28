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
        self.input_filter = nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                                        nn.ReLU(inplace=True))
        self.bias_filter = nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                                        nn.ReLU(inplace=True))
        self.filter = nn.Sequential(nn.Linear(in_features*2, out_features, bias=True),
                                        nn.Tanh())

    def forward(self, input, bias):
        output = torch.cat([self.input_filter(input), self.bias_filter(bias)], 1)
        return self.filter(output)


class ModuleNet(nn.Module):
    def __init__(self, alpha,
                 num_module,
                 embedding,
                 embed_size,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.alpha = alpha
        self.dropout_rate = 0.3

        # self.entity_embeds = embedding
        self.entity_embeds = Variable(torch.from_numpy(embedding).float(), requires_grad=False).cuda()

        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        # nn.Dropout(p=self.dropout_rate, inplace=True),
                                        nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

        self.function_modules = {}
        for id in range(num_module):
            module = ModuleBlock(embed_size, embed_size)
            self.add_module(str(id), module)
            self.function_modules[id] = module

    # def look_up_embed(self, id):
    #     lookup_tensor = torch.LongTensor([id]).cuda()
    #     return self.entity_embeds(autograd.Variable(lookup_tensor))
    #
    # def look_up_embeds(self, ids):
    #     lookup_tensor = torch.LongTensor(ids).cuda()
    #     return self.entity_embeds(autograd.Variable(lookup_tensor))
    #
    # def update_embed(self, id, new):
    #     self.entity_embeds.weight.data[id] = new.data

    def look_up_embed(self, id):
        return self.entity_embeds[id].view(1,-1)

    def look_up_embeds(self, ids):
        return self.entity_embeds[ids]

    def update_embed(self, id, new):
        self.entity_embeds.data[id] = new.data

    def forward_path(self, path):
        x = self.look_up_embed(path[0])
        length = len(path)
        for i in range(1, length, 2):
            module = self.function_modules[path[i]]
            bias = self.look_up_embed(path[i+1])
            x = module(x, bias)
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # x = F.normalize(x)
        # w = 1/(length*self.alpha)
        self.update_embed(path[-1], x)
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
