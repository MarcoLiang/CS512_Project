import argparse
parser = argparse.ArgumentParser()
import json
import numpy as np
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data_baseline import BaselineData


class EmbeddingTrainer(nn.Module):
    def __init__(self, num_entity,
                 embed_size,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(EmbeddingTrainer, self).__init__()

        self.embed = nn.Embedding(num_entity, embed_size)

        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

    def look_up_embed(self, id):
        lookup_tensor = torch.from_numpy(id).cuda()
        return self.embed(autograd.Variable(lookup_tensor))

    def forward(self, batch):
        data_output = self.look_up_embed(batch)
        output = self.classifier(data_output)
        return output

def train_embedding(data_dir, learning_rate, batch_size, embed_size, classifier_hidden_dim, classifier_output_dim, num_epoch):
    print("Starting training embedding...")

    dataset = BaselineData(data_dir)
    num_entity = dataset.author_num + dataset.bias_num

    kwargs = {
        'num_entity': num_entity,
        'embed_size': embed_size,
        'classifier_hidden_dim': classifier_hidden_dim,
        'classifier_output_dim': classifier_output_dim
    }

    model = EmbeddingTrainer(**kwargs)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    t = 0
    epoch = 0
    while epoch < num_epoch:
        dataset.shuffle()
        epoch += 1
        loss_aver = 0
        for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=batch_size):
            t += 1
            ids, labels = batch
            # label_var = Variable(torch.LongTensor(labels))
            label_var = Variable(torch.LongTensor(labels).cuda())
            optimizer.zero_grad()
            scores = model(ids)
            loss = loss_fn(scores, label_var)
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            val_acc = check_accuracy(dataset, model, batch_size)
            print('epoch:{} training accuracy is {}'.format(epoch, val_acc))

    print("training is done!")
    return model.embed, model.classifier


def check_accuracy(dataset, model, batch_size):
    model.eval()
    num_correct, num_samples = 0, 0
    for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=batch_size):
        ids, labels = batch
        scores = model(ids)
        preds = np.argmax(scores.data.cpu().numpy(), axis=1)
        num_correct += np.sum(preds == labels)
        num_samples += len(labels)
    valid_acc = float(num_correct) / num_samples
    model.train()
    return valid_acc


if __name__ == '__main__':
    pass