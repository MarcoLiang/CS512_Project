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

from utils.data_baseline import Data
from utils.load_embedding import *

# data options
parser.add_argument('--batch_size', default=128)
parser.add_argument('--data_dir', default="./data/classify_task_170W/pattern_30_70")

# module options
parser.add_argument('--embed_size', default=128)
# parser.add_argument('--embed', default='dw')
# parser.add_argument('--embed_path', default="./embedding_file/deepwalk/focus_embedding")
parser.add_argument('--embed', default='esim')
parser.add_argument('--embed_path', default="./embedding_file/esim/vec_dim_128.dat")
parser.add_argument('--id_path', default="./data/focus/venue_filtered_unique_id")
parser.add_argument('--classifier_hidden_dim', default=32)
parser.add_argument('--classifier_output_dim', default=4)

# Optimization options
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_epoch', default=1000)

# Output options
parser.add_argument('--checkpoint_path', default='./model/baseline_model_classification/checkpoint.pt')
parser.add_argument('--check_every', default=10)
parser.add_argument('--record_loss_every', default=1)


class BaselineMLP(nn.Module):
    def __init__(self, embed_mode,
                 embed_size,
                 embed_path,
                 id_path,
                 classifier_hidden_dim,
                 classifier_output_dim,
                 verbose=True):
        super(BaselineMLP, self).__init__()

        embed = embedding_loader(id_path, embed_path, embed_mode)
        self.entity_embeds = Variable(torch.from_numpy(embed).float(), requires_grad=False)

        self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))

    def look_up_entity_embed(self, id):
        return self.entity_embeds[id].view(1,-1)

    def forward(self, batch):
        input = []
        for id in batch:
            input.append(self.look_up_entity_embed(id))
        inputs = torch.cat(input, 0)
        output = self.classifier(inputs)
        return output


def main(args):
    # load data
    dataset = Data(args.data_dir)
    train_model(dataset, args)

def train_model(dataset, args):

    kwargs = {
        'embed_mode': args.embed,
        'embed_size': args.embed_size,
        'embed_path': args.embed_path,
        'id_path': args.id_path,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    execution_engine = BaselineMLP(**kwargs)
    # execution_engine.cuda()
    execution_engine.train()
    optimizer = torch.optim.Adam(execution_engine.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()#.cuda()

    stats = {
        'train_losses': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_es': [],
        'best_val_acc': -1, 'model_e': 0,
    }

    epoch = 0
    while epoch < args.num_epoch:
        dataset.shuffle()
        epoch += 1
        # print('Starting epoch %d' % epoch)
        # loss_aver = 0

        ids, labels = dataset.X_train, dataset.y_train
        label_var = Variable(torch.LongTensor(labels))
        optimizer.zero_grad()
        scores = execution_engine(ids)
        loss = loss_fn(scores, label_var)
        # loss_aver += loss.data[0]
        loss.backward()
        optimizer.step()

        if epoch % args.check_every == 0:
            print('Checking training/validation accuracy ... ')
            val_acc = check_accuracy(dataset, execution_engine, args.batch_size)
            print('val accuracy is ', val_acc)
            stats['val_accs'].append(val_acc)
            stats['val_accs_es'].append(epoch)

            if val_acc > stats['best_val_acc']:
                stats['best_val_acc'] = val_acc
                stats['model_e'] = epoch
                best_state = get_state(execution_engine)

            checkpoint = {
                'args': args,
                'kwargs': kwargs,
                'state': best_state,
                'stats': stats,
            }
            for k, v in stats.items():
                checkpoint[k] = v
            print('Saving checkpoint to %s' % args.checkpoint_path)
            torch.save(checkpoint, args.checkpoint_path)

    print("training is done!")
    print("best validate accuracy:{}".format(stats['best_val_acc']))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def check_accuracy(dataset, model, batch_size):
    model.eval()
    ids, labels = dataset.X_test, dataset.y_test
    scores = model(ids)
    preds = np.argmax(scores.data.numpy(), axis=1)
    num_correct = np.sum(preds == labels)
    valid_acc = float(num_correct) / len(dataset.y_test)
    model.train()
    return valid_acc


def load_cpu(path):
  """
  Loads a torch checkpoint, remapping all Tensors to CPU
  """
  return torch.load(path, map_location=lambda storage, loc: storage)


def load_model(path, verbose=True):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['kwargs']
  state = checkpoint['state']
  args = checkpoint['args']
  kwargs['verbose'] = verbose
  model = BaselineMLP(**kwargs)
  model.load_state_dict(state)
  return model, kwargs, args


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)