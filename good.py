import argparse
import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.data3 import Data
from model.good_model import ModuleNet
from run_baseline import BaselineMLP
from utils.data_baseline import BaselineData
from utils.load_embedding import *
from model.embedding_model import train_embedding
parser = argparse.ArgumentParser()
torch.backends.cudnn.enabled = True

# data options
parser.add_argument('--batch_size', default=128)
parser.add_argument('--data_dir', default="./data/classify_task/pattern_90_10")

# training embed options
parser.add_argument('--batch_size_for_embed', default=128)
parser.add_argument('--num_epoch_for_embed', default=100)

# module options
parser.add_argument('--alpha', default=3)
parser.add_argument('--embed_size', default=128)
parser.add_argument('--embed', default='dw')
parser.add_argument('--embed_path', default="./embedding_file/deepwalk/focus_embedding_new")
# parser.add_argument('--embed', default='esim')
# parser.add_argument('--embed_path', default="./embedding_file/esim/vec_dim_128.dat")
parser.add_argument('--id_path', default="./data/focus/venue_filtered_unique_id")
parser.add_argument('--classifier_hidden_dim', default=64)
parser.add_argument('--classifier_output_dim', default=4)

# Optimization options
parser.add_argument('--learning_rate', default=5e-5)
parser.add_argument('--num_epoch', default=100000)

# Output options
parser.add_argument('--checkpoint_path', default='./model/trained_model_classification_90_10/checkpoint.pt')
parser.add_argument('--check_every', default=1)
parser.add_argument('--record_loss_every', default=10000)


def train_embedding(dataset, embed, args):
    kwargs = {
        'embed': embed,
        'embed_size': args.embed_size,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    baseline_model = BaselineMLP(**kwargs).cuda()
    baseline_model.train()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    epoch = 0
    best_test_acc = 0
    best_train_acc = 0
    num_train = len(dataset.y_train)
    num_test = len(dataset.y_test)
    while epoch < 1000:
        dataset.shuffle()
        epoch += 1

        baseline_model.train()
        ids, labels = dataset.X_train, dataset.y_train
        label_var = Variable(torch.LongTensor(labels)).cuda()
        optimizer.zero_grad()
        scores = baseline_model(ids)
        loss = loss_fn(scores, label_var)
        loss.backward()
        optimizer.step()

        baseline_model.eval()
        ids, labels = dataset.X_test, dataset.y_test
        scores = baseline_model(ids)
        preds = np.argmax(scores.data.cpu().numpy(), axis=1)
        num_correct = np.sum(preds == labels)
        valid_acc = float(num_correct) / num_test
        best_test_acc = max(best_test_acc, valid_acc)

        ids, labels = dataset.X_train, dataset.y_train
        scores = baseline_model(ids)
        preds = np.argmax(scores.data.cpu().numpy(), axis=1)
        num_correct = np.sum(preds == labels)
        train_acc = float(num_correct) / num_train
        best_train_acc = max(best_train_acc, train_acc)

    return best_train_acc, best_test_acc


def train_model(args):
    # load data
    dataset = Data(args.data_dir)
    baseline_dataset = BaselineData(args.data_dir)
    args.num_module = dataset.nn_num
    # start_has_label = dataset.y_train[:,0] != -2
    # dataset.X_train = dataset.X_train[start_has_label]
    # dataset.y_train = dataset.y_train[start_has_label]
    num_test = len(dataset.y_test)
    num_train = len(dataset.y_train)
    num_node = dataset.bias_num + dataset.author_num
    no_label = dataset.y_train[:,-1] == -2
    has_label = np.invert(no_label)
    dataset.X_train = dataset.X_train[has_label]
    dataset.y_train = dataset.y_train[has_label]
    no_label = dataset.y_train[:, 0] == -2
    num_of_no_label_train = np.sum(no_label)
    num_of_has_label_train = len(dataset.y_train) - num_of_no_label_train
    # num_of_no_label_train = np.sum(no_label)
    # num_of_has_label_train = num_train - num_of_no_label_train




    # if num_of_has_label_train>num_of_no_label_train:
    #     ratio_for_has_label = num_of_no_label_train/num_of_has_label_train
    #     ratio_for_no_label = 1
    # else:
    #     ratio_for_no_label = num_of_has_label_train / num_of_no_label_train
    #     ratio_for_has_label = 1
    # unique_label, label_counts = np.unique(dataset.y_train[has_label][:, -1], return_counts=True)
    # min_count = np.min(label_counts)
    # ratio_label = min_count/label_counts
    # print('train label {}\tcounts {}'.format(unique_label, label_counts))
    # unique_label, label_counts = np.unique(dataset.y_test, return_counts=True)
    # print('test label {}\tcounts {}'.format(unique_label, label_counts))

    print('num of node', num_node)
    print('num of author in training set', dataset.author_num - num_test)
    print('num of author in test ser', num_test)
    print('num of pathes in training set', num_train)
    print('num of pathes in training set with labeled start', num_of_has_label_train)
    print('num of pathes in training set without labeled start', num_of_no_label_train)

    embed_kwargs = {
        'data_dir': args.data_dir,
        'learning_rate': args.learning_rate,
        'embed_size': args.embed_size,
        'batch_size': args.batch_size_for_embed,
        'num_epoch': args.num_epoch_for_embed,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    # decay_rate = 0.95
    # decay_time = 10000

    # embed = nn.Embedding(num_node, args.embed_size)
    # embed.weight.data.copy_(torch.from_numpy(embedding_loader(args.id_path, args.embed_path, args.embed)))

    # embed, _ = train_embedding(**embed_kwargs)
    # embed = embed.weight.data.cpu().numpy()

    # embed = nn.Embedding(num_node, args.embed_size)

    # embed = embedding_loader(args.id_path, args.embed_path, args.embed)
    # print('Creating embedding...', num_node, args.embed_size)

    kwargs = {
        # 'embedding': embed,
        'num_author': dataset.author_num,
        'num_node': num_node,
        'embed_size': args.embed_size,
        'num_module': args.num_module,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_output_dim': args.classifier_output_dim
    }

    execution_engine = ModuleNet(**kwargs)
    execution_engine.cuda()
    execution_engine.train()
    # optimizer_for_all = optim.Adam(execution_engine.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam([{'params':execution_engine.module_params, 'lr':5e-3},
    #                               {'params':execution_engine.classifier.parameters(), 'lr':5e-4},
    #                                {'params':execution_engine.author_embeds.parameters(),'lr':5e-3}], lr=args.learning_rate)
    optimizer = optim.Adam(execution_engine.parameters(), lr=args.learning_rate)
    # optimizer_for_embed = optim.Adam(execution_engine.author_embeds.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    # stats = {
    #     'train_losses': [], 'train_losses_ts': [], 'train_losses_ms': [],
    #     'train_accs': [], 'val_accs': [], 'val_accs_es': [],
    #     'best_val_acc': -1, 'model_e': 0,
    # }

    t = 0
    m = 0
    epoch = 0

    author = {}
    author_path = {}
    for i in range(dataset.author_num):
        author[i] = []
    for i, path in enumerate(dataset.X_train):
        author[path[0]].append(i)
    for author, ids in author.items():
        five = []
        seven = []
        nine = []
        for id in ids:
            if len(dataset.X_train[id]) == 5:
                five.append(id)
            elif len(dataset.X_train[id]) == 7:
                seven.append(id)
            else:
                nine.append(id)
        author_path[author] = (five, seven, nine)




    print("Starting training our model...")
    while epoch < args.num_epoch:

        train = []
        for author, path in author_path.items():
            five, seven, nine = path
            tmp = 2000
            if bool(five):
                five_choice = np.random.choice(five, size=1000)
            else:
                tmp+=1000
                five_choice=[]
            if bool(seven):
                seven_choice = np.random.choice(seven, size=1000)
            else:
                tmp+=1000
                seven_choice=[]
            nine_choice = np.random.choice(nine, size=tmp)
            train += np.concatenate([five_choice, seven_choice, nine_choice]).tolist()
        random_choice = np.array(train, dtype=int)
        np.random.shuffle(random_choice)

        dataset.shuffle()
        epoch += 1
        print('='*50)
        print('Starting epoch %d' % epoch)
        loss_aver = 0
        # optimizer = optimizer_for_all
        # flag = 1
        # counts_for_label = np.zeros(4)

        # random_choice = np.random.choice(range(num_of_has_label_train), size=args.record_loss_every*args.batch_size)
        # for batch in dataset.next_batch(dataset.X_train[random_choice], dataset.y_train[random_choice], batch_size=args.batch_size):
        #     paths, labels = batch
        #     labels = labels[:,-1]
        #     label_var = Variable(torch.LongTensor(labels).cuda())
        #     optimizer_for_all.zero_grad()
        #     scores = execution_engine(paths)
        #     loss = loss_fn(scores, label_var)
        #     loss_aver += loss.data[0]
        #     loss.backward()
        #     optimizer_for_all.step()
        #
        # loss_aver /= args.record_loss_every
        # embedding = execution_engine.author_embeds.weight.data.cpu().numpy()
        # best_train_acc, best_test_acc = train_embedding(baseline_dataset, embedding, args)
        # print('training whole para: loss={} embed: train={} test={}'.format(loss_aver, best_train_acc,best_test_acc))
        # loss_aver = 0

        # print('start traning for embed:')
        for batch in dataset.next_batch(dataset.X_train[random_choice], dataset.y_train[random_choice], batch_size=args.batch_size):
            paths, labels = batch
            # for i in range(len(labels)):
            #     path, label = paths[i], labels[i, -1]
            #     if label == -2:
            #         continue
            #         # if np.random.random() > ratio_for_no_label:
            #         #     continue
            #         # m += 1
            #         # execution_engine.forward_path(path)
            #     else:
            #         # if np.random.random() > ratio_label[label]:
            #         # if np.random.random() > ratio_for_has_label:
            #         #     continue
            #         t += 1
            #         # counts_for_label[label] += 1
            #         label_var = Variable(torch.LongTensor([int(label)]).cuda())
            #         optimizer.zero_grad()
            #         scores = execution_engine([path])
            #         loss = loss_fn(scores, label_var)
            #         loss_aver += loss.data[0]
            #         loss.backward()
            #         optimizer.step()

            t += 1
            labels = labels[:,-1]
            label_var = Variable(torch.LongTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, label_var)
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()

            if t % args.record_loss_every == 0:
                # if flag:
                #     optimizer = optimizer_for_embed
                #     flag = 0
                # else:
                #     optimizer = optimizer_for_all
                #     flag = 1
                # print('counts for each label', counts_for_label)
                # counts_for_label = np.zeros(4)
                loss_aver /= args.record_loss_every
                # acc = check_accuracy(dataset, execution_engine)
                embedding = execution_engine.author_embeds.weight.data
                best_train_acc, best_test_acc = train_embedding(baseline_dataset, embedding, args)
                print('{} {} loss={} embed: train={} test={}'.format(t,m,loss_aver,best_train_acc,best_test_acc))
                # stats['train_losses'].append(loss_aver)
                # stats['train_losses_ts'].append(t)
                # stats['train_losses_ms'].append(m)
                loss_aver = 0

                    # if t % decay_time == 0:
                    #     execution_engine.w = execution_engine.w*decay_rate

        # if epoch % args.check_every == 0:
        #     print('Checking training/validation accuracy ... ')
        #     val_acc = check_accuracy(dataset, execution_engine, 64)
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
    # print("best validate accuracy:{}".format(stats['best_val_acc']))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def check_accuracy(dataset, model):
    model.eval()

    ids = dataset.X_test
    labels = dataset.y_test
    scores = model.predict(ids.tolist())
    preds = np.argmax(scores.data.cpu().numpy(), axis=1)
    correct = preds[preds == labels]
    # print(preds)
    # print(labels)
    num_correct = np.sum(preds == labels)
    num_samples = len(labels)
    valid_acc = float(num_correct) / num_samples
    labels, counts = np.unique(preds, return_counts=True)
    print('prediced labels', labels, 'counts', counts)
    labels, counts = np.unique(correct, return_counts=True)
    print('correct labels', labels, 'counts', counts)
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