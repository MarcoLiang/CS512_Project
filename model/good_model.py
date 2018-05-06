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

# class ModuleBlock(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(ModuleBlock, self).__init__()
#         self.GRU = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=True)
#
#     def forward(self, input, bias):
#         return self.GRU(input, bias)

#
# class ModuleBlock(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ModuleBlock, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features))
#         self.bias_filter = nn.Sequential(nn.Linear(2*in_features, out_features, bias=True),
#                                          nn.Tanh())
#                                         # nn.Dropout(p=self.dropout_rate, inplace=True),
#                                         # nn.Linear(in_features, out_features, bias=True))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, bias=None, flag=True):
#         # return F.linear(input, self.weight, bias)
#     #     if flag:
#     #         return F.linear(input, self.weight, bias)
#     #     else:
#     #         return F.linear(input, self.weight, self.bias)
#         # else:
#         #     return F.relu(F.linear(input, self.weight, self.bias), inplace=True)
#         if flag:
#             return F.linear(input, self.weight, self.bias_filter(bias))
#         else:
#             return F.linear(input, self.weight, self.bias)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'in_features=' + str(self.in_features) \
#             + ', out_features=' + str(self.out_features)


class ModuleNet(nn.Module):
    def __init__(self,
                 num_node,
                 num_author,
                 num_module,
                 # embedding,
                 embed_size,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.dropout_rate = 0.2
        self.num_author = num_author
        self.num_node = num_node

        # self.entity_embeds = embedding
        # self.entity_embeds = Variable(torch.from_numpy(embedding).float(), requires_grad=False).cuda()

        self.author_embeds = nn.Embedding(num_author, embed_size)
        # self.author_embeds.weight.data.copy_(torch.from_numpy(embedding[:num_author]))
        # self.node_embeds = Variable(torch.from_numpy(embedding).float(), requires_grad=False).cuda()
        # word_embeds = np.load('./embedding_file/w2v/w2vembed_128.npy')
        # self.word_embeds = Variable(torch.from_numpy(word_embeds).float(), requires_grad=False).cuda()

        # self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
        #                                 nn.Tanh(),
        #                                 # nn.Dropout(p=self.dropout_rate, inplace=True),
        #                                 nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_output_dim, bias=True))

        # self.module_params = []
        self.function_modules = {}
        for id in range(num_module):
            # module = ModuleBlock(embed_size, embed_size)
            module = nn.Sequential(nn.Linear(in_features=embed_size, out_features=embed_size, bias=True))
            self.add_module(str(id), module)
            self.function_modules[id] = module
            # self.module_params.append(module.weight)
            # self.module_params.append(module.bias)

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

    # def look_up_embed(self, id):
    #     return self.entity_embeds[id].view(1,-1)
    #
    # def look_up_embeds(self, ids):
    #     return self.entity_embeds[ids]
    #
    # def update_embed(self, id, new):
    #     self.entity_embeds.data[id] = new.data

    def look_up_embed(self, id):
        lookup_tensor = torch.LongTensor([id]).cuda()
        return self.author_embeds(autograd.Variable(lookup_tensor))

    def look_up_node_embed(self, id):
        return self.node_embeds[id].view(1,-1)

    def look_up_word_embed(self, id):
        return self.word_embeds[id-self.num_author].view(1,-1)

    def look_up_embeds(self, ids):
        lookup_tensor = torch.LongTensor(ids).cuda()
        return self.author_embeds(autograd.Variable(lookup_tensor))

    def update_embed(self, id, new):
        self.author_embeds.weight.data[id] = new.data

    def forward_path(self, path):
        # x = self.look_up_node_embed(path[0])
        x = self.look_up_embed(path[0])
        length = len(path)
        for i in range(1, length-2, 2):
            module = self.function_modules[path[i]]
            # bias = self.look_up_node_embed(path[i+1])
            # bias1 = self.look_up_node_embed(path[i + 1])
            # bias2 = self.look_up_word_embed(path[i + 1])
            # bias = torch.cat([bias1, bias2], 1)
            x = F.tanh(module(x))
            # x = F.tanh(module(x, bias, flag=True))
            # x = F.tanh(module(x, bias))
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        module = self.function_modules[path[-2]]
        # bias = self.look_up_node_embed(path[-1])
        x = module(x)
        # x = module(x, bias)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # module = self.function_modules[path[-2]]
        # bias = self.look_up_embed(path[-1])
        # x = F.tanh(module(x, flag=False))
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # w = 1/(length*self.alpha)
        # old = self.look_up_embed(path[-1])
        # output = (1-self.w) * old + self.w * x
        # self.update_embed(path[-1], output)
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
