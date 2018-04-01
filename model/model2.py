import numpy as np
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModuleNet(nn.Module):
    def __init__(self, num_entity,
                 num_node,
                 embed_size,
                 classifier_first_dim,
                 classifier_second_dim,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.embeds = nn.Embedding(num_entity, embed_size)
        self.function_modules = {}
        for id in range(1, num_node+1):
            module = nn.Linear(embed_size, embed_size, bias=True)
            self.add_module(str(id), module)
            self.function_modules[id] = module
        self.classifier = nn.Sequential(nn.Linear(2*embed_size, classifier_first_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_first_dim, classifier_second_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_second_dim, 1, bias=True),
                                        nn.Sigmoid())

    def look_up_embed(self, id):
        lookup_tensor = torch.LongTensor([id-1]).cuda()
        return self.embeds(autograd.Variable(lookup_tensor))

    def forward_path(self, path):
        current_id = path[0]
        x = self.look_up_embed(current_id)
        for i in path[1:-1]:
            module = self.function_modules[i]
            x = module(x)
        current_id = path[-1]
        y = self.look_up_embed(current_id)
        for i in path[1:-1]:
            module = self.function_modules[i]
            y = module(y)
        return torch.cat([x, y], 1)

    def forward(self, batch):
        output = []
        for data in batch:
            total, data_output = 0, 0
            for path in data:
                count = path[-1]
                total += count
                data_output += count * self.forward_path(path[:-1])
            data_output /= total
            output.append(self.classifier(data_output))
        output =torch.cat(output, 0)
        return output
