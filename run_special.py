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
parser.add_argument('--batch_size', default=2048)
parser.add_argument('--data_dir', default="./data/classify_task_170W/pattern_30_70")

# training embed options
parser.add_argument('--batch_size_for_embed', default=64)
parser.add_argument('--num_epoch_for_embed', default=100)

# module options
parser.add_argument('--threshold', default=0.6)
parser.add_argument('--alpha', default=3)
parser.add_argument('--embed_size', default=128)
parser.add_argument('--classifier_hidden_dim', default=32)
parser.add_argument('--classifier_output_dim', default=4)

# Optimization options
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_epoch', default=100000)

# Output options
parser.add_argument('--save_path', default='./model/no_embedding_model_classification/')
parser.add_argument('--check_every', default=1)
parser.add_argument('--record_loss_every', default=100)


def train_model(args):
    args.checkpoint_path = args.save_path+'checkpoint.pt'
    args.log_path = args.save_path+'log.txt'

    embed_kwargs = {
        'data_dir': args.data_dir,
        'learning_rate': args.learning_rate,
        'embed_size': args.embed_size,
        'batch_size': args.batch_size_for_embed,
        'num_epoch': args.num_epoch_for_embed,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    dataset = Data(args.data_dir)
    entity2label = dict(zip(dataset.X_test, dataset.y_test))
    num_test = len(dataset.y_test)
    num_train = len(dataset.y_train)
    num_node = dataset.bias_num + dataset.author_num
    print('num of node', num_node)
    print('num of author in training set', dataset.author_num-num_test)
    print('num of author in test ser', num_test)
    print('num of pathes in training set', num_train)

    # embed, classifier = train_embedding(**embed_kwargs)
    embed = nn.Embedding(num_node, args.embed_size)
    print('Creating embedding...', num_node, args.embed_size)

    # args.alpha = len(dataset.y_train)/len(dataset.y_test)

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

    # for i in execution_engine.parameters():
    #     print(i)

    stats = {
        'train_losses': [], 'train_losses_ts': [], 'train_losses_ms': [],
        'train_accs': [], 'val_accs': [], 'val_accs_es': [],
        'best_val_acc': -1, 'model_e': 0,
    }

    log = open(args.log_path, 'w')
    t = 0
    m = 0
    epoch = 0
    print("Starting training our model...")
    while epoch < args.num_epoch:

        dataset.shuffle()
        epoch += 1
        print('Starting epoch %d' % epoch)
        loss_aver = 0
        num_correct1, num_correct2 = 0, 0
        num_sample1, num_sample2 = 0, 0
        entity2preds = dict(zip(dataset.X_test, np.zeros((num_test, args.classifier_output_dim))))

        for batch in dataset.next_batch(dataset.X_train, dataset.y_train, batch_size=args.batch_size):

            t += 1
            paths, labels = batch
            labels = labels[:,-1]
            p = labels!=-2
            paths1 = paths[p]
            labels1 = labels[p]
            label_var = Variable(torch.LongTensor(labels1).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths1)
            loss = loss_fn(scores, label_var)
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()
            scores = scores.data.cpu().numpy()
            preds = np.argmax(scores, axis=1)
            num_correct1 += np.sum(preds == labels1)
            num_sample1 += len(labels1)

            q = labels==-2
            paths2 = paths[q]
            test = [path[-1] for path in paths2]
            labels2 = np.array([entity2label[i] for i in test])
            scores = F.softmax(execution_engine(paths2), dim=1)
            scores = scores.data.cpu().numpy()
            preds = np.argmax(scores, axis=1)
            num_correct2 += np.sum(preds == labels2)
            num_sample2 += len(labels2)

            for i in range(len(labels2)):
                pred = preds[i]
                if scores[i][pred] > args.threshold:
                    entity2preds[test[i]][pred] += 1

            num_test = len(dataset.y_test)

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

        train_acc = float(num_correct1) / num_sample1
        print('train path accuracy is ', train_acc)
        test_acc = float(num_correct2) / num_sample2
        print('test path accuracy is ', test_acc)

        en_results = np.array([entity2preds[i] for i in dataset.X_test])
        max_votes = np.max(en_results, axis=1)
        votes = np.sum(en_results, axis=1)
        votes_rate = max_votes/votes
        np.nan_to_num(votes_rate, copy=False)
        en_preds = np.argmax(en_results, axis=1)
        en_correct = en_preds==dataset.y_test
        en_acc = float(np.sum(en_correct)) / num_test
        print('ensemble acc is ', en_acc)
        vrc = votes_rate[en_correct]
        vrc_mean = np.mean(vrc)
        vrc_min = np.min(vrc)
        vrc_max = np.max(vrc)
        vrw = votes_rate[np.invert(en_correct)]
        vrw_mean = np.mean(vrw)
        vrw_min = np.min(vrw)
        vrw_max = np.max(vrw)
        print('vote rate for correct mean={} min={} max={}'.format(vrc_mean, vrc_min, vrc_max))
        print('vote rate for wrong mean={} min={} max={}'.format(vrw_mean, vrw_min, vrw_max))

        # threshold = 0.9
        # sure_answers = votes_rate>threshold
        # sure_ids = dataset.X_test[sure_answers]
        # sure_preds = np.argmax(en_results[sure_answers])
        # ids2preds = dict(zip(sure_ids, sure_preds))
        # for i, path in enumerate(dataset.X_train):
        #     last = path[-1]
        #     if last in sure_ids:
        #         dataset.y_train[i][-1] = ids2preds[last]


        log.write('epoch:{}\ntrain path acc={}\ttest path acc={}\tensemble acc={}'
                  '\nvote rate for correct mean={} max={} min={}'
                  '\nvote rate for wrong mean={} max={} min={}\n'
                  .format(epoch, train_acc, test_acc, en_acc, vrc_mean, vrc_max, vrc_min, vrw_mean,vrw_max, vrw_min))
        log.write('id\tresults\t\t\t\tlabel\tcorrect\tvote rate\n')
        log_info = np.concatenate((dataset.X_test.reshape(num_test,1), en_results, dataset.y_test.reshape(num_test,1), en_correct.reshape(num_test,1), votes_rate.reshape(num_test,1)), axis=1)
        log_info = sorted(log_info, key=lambda x:x[-2])
        for i in range(num_test):
            log.write('\t'.join([str(j) for j in log_info[i]])+'\n')

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