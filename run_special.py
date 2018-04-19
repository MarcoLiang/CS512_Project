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

from utils.data3 import Data
from model.model5 import ModuleNet
from model.embedding_model import train_embedding


# data options
parser.add_argument('--batch_size', default=512)
parser.add_argument('--data_dir', default="./data/classify_task/pattern_30_70")

# training embed options
parser.add_argument('--batch_size_for_embed', default=64)
parser.add_argument('--num_epoch_for_embed', default=100)

# module options
parser.add_argument('--alpha', default=3)
parser.add_argument('--embed_size', default=128)
parser.add_argument('--classifier_hidden_dim', default=32)
parser.add_argument('--classifier_output_dim', default=4)

# Optimization options
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_epoch', default=100000)

# Output options
parser.add_argument('--checkpoint_path', default='./model/no_embedding_model_classification/checkpoint.pt')
parser.add_argument('--check_every', default=1)
parser.add_argument('--record_loss_every', default=100)


def train_model(args):

    embed_kwargs = {
        'data_dir': args.data_dir,
        'learning_rate': args.learning_rate,
        'embed_size': args.embed_size,
        'batch_size': args.batch_size_for_embed,
        'num_epoch': args.num_epoch_for_embed,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    embed, classifier = train_embedding(**embed_kwargs)

    dataset = Data(args.data_dir)


    alpha = len(dataset.y_train)/len(dataset.y_test)

    args.num_module = dataset.nn_num
    kwargs = {
        'alpha': args.alpha,
        'embedding': embed,
        'embed_size': args.embed_size,
        'num_module': args.num_module,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    execution_engine = ModuleNet(**kwargs)
    execution_engine.cuda()
    execution_engine.train()
    optimizer = torch.optim.Adam(execution_engine.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    stats = {
        'train_losses': [], 'train_losses_ts': [], 'train_losses_ms': [],
        'train_accs': [], 'val_accs': [], 'val_accs_es': [],
        'best_val_acc': -1, 'model_e': 0,
    }

    t = 0
    m = 0
    epoch = 0
    print("Starting training our model...")
    while epoch < args.num_epoch:
        dataset.shuffle()
        epoch += 1
        print('Starting epoch %d' % epoch)
        loss_aver = 0
        num_correct, num_samples = 0, 0
        for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=args.batch_size):
            t += 1
            paths, labels = batch
            labels = labels[:,-1]
            p = labels!=-2
            paths = paths[p]
            labels = labels[p]
            label_var = Variable(torch.LongTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, label_var)
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()
            preds = np.argmax(scores.data.cpu().numpy(), axis=1)
            num_correct += np.sum(preds == labels)
            num_samples += len(labels)
            # for i in range(len(labels)):
            #     path, label = paths[i], labels[i, -1]
            #     if label == -2:
            #         continue
            #         if np.random.random() > alpha:
            #             continue
            #         m += 1
            #         execution_engine.forward_path(path)
            #     else:
            #         t += 1
            #         label_var = Variable(torch.LongTensor([int(label)]).cuda())
            #         optimizer.zero_grad()
            #         scores = execution_engine([path])
            #         loss = loss_fn(scores, label_var)
            #         loss_aver += loss.data[0]
            #         loss.backward()
            #         optimizer.step()


            if t % args.record_loss_every == 0:
                loss_aver /= args.record_loss_every
                print(t, m, loss_aver)
                stats['train_losses'].append(loss_aver)
                stats['train_losses_ts'].append(t)
                stats['train_losses_ms'].append(m)
                loss_aver = 0

        acc = float(num_correct) / num_samples
        print('accuracy is ', acc)

        # if epoch % args.check_every == 0:
        #     print('Checking training/validation accuracy ... ')
        #     val_acc = check_accuracy(dataset, execution_engine, args.batch_size)
        #     print('val accuracy is ', val_acc)
        #     stats['val_accs'].append(val_acc)
        #     stats['val_accs_es'].append(epoch)
        #
        #     if val_acc > stats['best_val_acc']:
        #         stats['best_val_acc'] = val_acc
        #         stats['model_e'] = epoch
        #         best_state = get_state(execution_engine)
        #
        #     checkpoint = {
        #         'args': args,
        #         'kwargs': kwargs,
        #         'state': best_state,
        #         'stats': stats,
        #     }
        #     for k, v in stats.items():
        #         checkpoint[k] = v
        #     print('Saving checkpoint to %s' % args.checkpoint_path)
        #     torch.save(checkpoint, args.checkpoint_path)

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
    for batch in dataset.next_batch(dataset.X_test, dataset.y_test, batch_size=batch_size):
        ids, labels = batch
        scores = model.predict(ids)
        preds = np.argmax(scores.data.cpu().numpy(), axis=1)
        num_correct += np.sum(preds == labels)
        num_samples += len(labels)
    valid_acc = float(num_correct) / num_samples

    model.train()
    # return train_acc, valid_acc
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
  model = ModuleNet(**kwargs)
  model.load_state_dict(state)
  return model, kwargs, args


def go_on(dataset, path, args):
    execution_engine, kwargs, _ = load_model(path)

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
        dataset.shuffle()
        epoch += 1
        print('Starting epoch %d' % epoch)
        loss_aver = 0
        for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=args.batch_size):
            t += 1
            paths, labels = batch
            labels_var = Variable(torch.FloatTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, labels_var.view(-1, 1))
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()

            if t % args.record_loss_every == 0:
                loss_aver /= args.record_loss_every
                print(t, loss_aver)
                stats['train_losses'].append(loss_aver)
                stats['train_losses_ts'].append(t)
                loss_aver = 0

        if epoch % args.check_every == 0:
            print('Checking training/validation accuracy ... ')
            # train_acc, val_acc = check_accuracy(dataset, execution_engine, args.batch_size)
            # print('train accuracy is', train_acc)
            # stats['train_accs'].append(train_acc)
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
                'stats':stats,
            }
            for k, v in stats.items():
                checkpoint[k] = v
            print('Saving checkpoint to %s' % args.checkpoint_path)
            torch.save(checkpoint, args.checkpoint_path)
            # del checkpoint['state']
            # with open(args.checkpoint_path + '.json', 'w') as f:
            #     json.dump(checkpoint, f)

    print("training is done!")
    print("best validate accuracy:{}".format(stats['best_val_acc']))


if __name__ == '__main__':
    args = parser.parse_args()

    train_model(args)

    # load data
    # dataset = Data()
    # # dataset.data_load(args.data_dir, args.max_length)
    # dataset.load_data("./data/pattern/meta_path_l1_new.txt", "./data/pattern/meta_path_l1_new_cnt.txt")
    # dataset.split_dataset()
    #
    # args.num_module = dataset.nn_num
    # args.num_bias = dataset.bias_num
    # args.num_entity = dataset.author_num
    #
    # go_on(dataset, args.checkpoint_path, args)