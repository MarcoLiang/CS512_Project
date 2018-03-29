import numpy as np
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ModuleBlock(nn.Module):
    def __init__(self, embed_size):
        super(ModuleBlock, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, embed_size).cuda(), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(1, embed_size).cuda(), requires_grad=True)
        torch.nn.init.xavier_uniform(self.weight)
        torch.nn.init.xavier_uniform(self.bias)

    def forward(self, x, y):
        # x, y are embeddings
        out = x * self.weight * y + self.bias
        out = F.relu(out)
        return out


class ModuleNet(nn.Module):
    def __init__(self, num_entity,
                 num_metapath,
                 max_length,
                 embed_size,
                 classifier_first_dim,
                 classifier_second_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.max_length = max_length
        self.embeds = nn.Embedding(num_entity, embed_size)
        self.function_modules = {}
        for id in range(1, num_metapath+1):
            module = ModuleBlock(embed_size=embed_size)
            self.add_module(str(id), module)
            self.function_modules[id] = module
        self.classifier = nn.Sequential(nn.Linear(2*embed_size*max_length, classifier_first_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_first_dim, classifier_second_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_second_dim, 1, bias=True),
                                        nn.Sigmoid())
        print(self.classifier)

    def look_up_embed(self, id):
        lookup_tensor = torch.LongTensor([id-1]).cuda()
        return self.embeds(autograd.Variable(lookup_tensor))

    def forward_path(self, path, length):
        current_id = path[0]
        x = self.look_up_embed(current_id)
        for i in range(length):
            next_id = path[2*(i+1)]
            y = self.look_up_embed(next_id)
            mid = path[2*i+1]
            module = self.function_modules[mid]
            x = module(x, y)
        output1 = x
        current_id = path[-1]
        x = self.look_up_embed(current_id)
        for i in reversed(range(length)):
            mid = path[2*i+1]
            module = self.function_modules[mid]
            next_id = path[2*i]
            y = self.look_up_embed(next_id)
            x = module(x, y)
        output2 = x
        return torch.cat([output1, output2], 1)

    def forward(self, batch):
        output = []
        for data in batch:
            assert len(data) <= self.max_length
            length = 0
            data_output = []
            for group in data:
                length += 1
                total = 0
                group_output = 0
                for path in group:
                    count = path[-1]
                    total += count
                    path = path[:-1]
                    path_output = count * self.forward_path(path, length)
                    group_output += path_output
                group_output /= total
                data_output.append(group_output)
            data_output = torch.cat(data_output, 1)
            output.append(self.classifier(data_output))
        output =torch.cat(output, 0)
        return output
