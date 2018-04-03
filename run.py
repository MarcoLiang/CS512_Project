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

from utils.data import Data
from model.model2 import ModuleNet

# data options
# parser.add_argument('--max_length', default=1)
parser.add_argument('--batch_size', default=64)
# parser.add_argument('--data_dir', default="./data/pattern/meta_path_l1_new.txt")

# module options
parser.add_argument('--embed_size', default=128)
# parser.add_argument('--classifier_first_dim', default=128)
# parser.add_argument('--classifier_second_dim', default=32)

# Optimization options
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_epoch', default=100000)

# Output options
parser.add_argument('--checkpoint_path', default='./model/trained_model/checkpoint.pt')
parser.add_argument('--check_every', default=1)
parser.add_argument('--record_loss_every', default=1)


def main(args):
    # load data
    dataset = Data()
    # dataset.data_load(args.data_dir, args.max_length)
    dataset.load_data("./data/pattern/meta_path_l1_new.txt", "./data/pattern/meta_path_l1_new_cnt.txt")

    for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=args.batch_size):
        paths, labels = batch

    args.num_metapath = dataset.nn_num
    args.num_bias = dataset.bias_num
    args.num_entity = dataset.author_num

    train_model(dataset, args)


def train_model(dataset, args):

    kwargs = {
        'embed_size': args.embed_size,
        'num_entity': args.num_entity,
        'num_metapath': args.num_metapath,
        'num_bias': args.num_bias,
        # 'max_length': args.max_length,
        # 'classifier_first_dim': args.classifier_first_dim,
        # 'classifier_second_dim': args.classifier_second_dim
    }

    execution_engine = ModuleNet(**kwargs)
    execution_engine.cuda()
    execution_engine.train()
    optimizer = torch.optim.Adam(execution_engine.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss().cuda()

    stats = {
        'train_losses': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_es': [],
        'best_val_acc': -1, 'model_e': 0,
    }

    t = 0
    epoch = 0
    while epoch < args.num_epoch:
        epoch += 1
        print('Starting epoch %d' % epoch)

        for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=args.batch_size):
            t += 1
            paths, labels = batch
            labels_var = Variable(torch.FloatTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, labels_var.view(-1, 1))
            loss.backward()
            # for param in execution_engine.parameters():
            #     print(param.data)
            #     print(param.grad.data.sum())
            optimizer.step()

            if t % args.record_loss_every == 0:
                print(t, loss.data[0])
                stats['train_losses'].append(loss.data[0])
                stats['train_losses_ts'].append(t)

        if epoch % args.check_every == 0:
            print('Checking training/validation accuracy ... ')
            train_acc, val_acc = check_accuracy(dataset, execution_engine, args.batch_size)
            print('train accuracy is', train_acc)
            print('val accuracy is ', val_acc)
            stats['train_accs'].append(train_acc)
            stats['val_accs'].append(val_acc)
            stats['val_accs_es'].append(epoch)

            if val_acc > stats['best_val_acc']:
                stats['best_val_acc'] = val_acc
                stats['model_e'] = epoch
                best_state = get_state(execution_engine)

            checkpoint = {
                'args': args.__dict__,
                'kwargs': kwargs,
                'state': best_state,
            }
            for k, v in stats.items():
                checkpoint[k] = v
            print('Saving checkpoint to %s' % args.checkpoint_path)
            torch.save(checkpoint, args.checkpoint_path)
            del checkpoint['state']
            with open(args.checkpoint_path + '.json', 'w') as f:
                json.dump(checkpoint, f)

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

    num_correct, num_samples = 0, 0
    for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=batch_size):
        paths, labels = batch
        labels_var = torch.FloatTensor(labels)
        scores = model(paths)
        preds = (scores.data>0.5).cpu().float().squeeze()
        num_correct += (preds == labels_var).sum()
        num_samples += preds.size(0)
    train_acc = float(num_correct) / num_samples

    num_correct, num_samples = 0, 0
    for batch in dataset.next_batch(dataset.X_valid, dataset.y_valid, batch_size=batch_size):
        paths, labels = batch
        labels_var = torch.FloatTensor(labels)
        scores = model(paths)
        preds = (scores.data > 0.5).cpu().float().squeeze()
        num_correct += (preds == labels_var).sum()
        num_samples += preds.size(0)
    valid_acc = float(num_correct) / num_samples

    model.train()
    return train_acc, valid_acc


def load_cpu(path):
  """
  Loads a torch checkpoint, remapping all Tensors to CPU
  """
  return torch.load(path, map_location=lambda storage, loc: storage)


def load_model(path, verbose=True):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['kwargs']
  state = checkpoint['state']
  kwargs['verbose'] = verbose
  model = ModuleNet(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
