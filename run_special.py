import argparse
import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.load_embedding import *
from utils.data3 import Data
from utils.data_baseline import BaselineData
from model.model5 import ModuleNet
from model.embedding_model import train_embedding
from run_baseline import BaselineMLP
parser = argparse.ArgumentParser()
torch.backends.cudnn.enabled = True

# data options
parser.add_argument('--batch_size', default=128)
parser.add_argument('--data_dir', default="./data/classify_task/pattern_90_10")

# training embed options
parser.add_argument('--batch_size_for_embed', default=64)
parser.add_argument('--num_epoch_for_embed', default=100)

# module options
parser.add_argument('--threshold', default=0.8)
parser.add_argument('--alpha', default=3)
parser.add_argument('--embed_size', default=128)
parser.add_argument('--embed', default='dw')
parser.add_argument('--embed_path', default="./embedding_file/deepwalk/focus_embedding_new")
# parser.add_argument('--embed', default='esim')
# parser.add_argument('--embed_path', default="./embedding_file/esim/vec_128_new.dat")
parser.add_argument('--id_path', default="./data/focus/venue_filtered_unique_id")
parser.add_argument('--classifier_hidden_dim', default=64)
parser.add_argument('--classifier_output_dim', default=4)

# Optimization options
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_epoch', default=100000)

# Output options
parser.add_argument('--save_path', default='./model/no_embedding_model_classification/')
parser.add_argument('--check_every', default=1)
parser.add_argument('--record_loss_every', default=200)


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
    loss_fn = torch.nn.CrossEntropyLoss()

    print('='*80)
    print('evaluate the embedding...')
    epoch = 0
    best_test_acc = 0
    best_train_acc = 0
    num_train = len(dataset.y_train)
    num_test = len(dataset.y_test)
    while epoch < 1500:
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


    print('best train acc = {}\tbest test acc = {}'.format(best_train_acc, best_test_acc))
    print('end the evaluation.')
    print('=' * 80)


def train_model(args):
    args.checkpoint_path = args.save_path+'checkpoint.pt'
    args.log_path = args.save_path+'log.txt'
    log = open(args.log_path, 'w')

    # baseline_dataset = BaselineData(args.data_dir)

    dataset = Data(args.data_dir)
    entity2label = dict(zip(dataset.X_test, dataset.y_test))
    num_test = len(dataset.y_test)
    num_train = len(dataset.y_train)
    num_node = dataset.bias_num + dataset.author_num
    print('num of node', num_node)
    print('num of author in training set', dataset.author_num-num_test)
    print('num of author in test ser', num_test)
    print('num of pathes in training set', num_train)

    # no_label = dataset.y_train[:,-1] == -2
    # num_of_no_label_train = np.sum(no_label)
    # num_of_has_label_train = num_train-num_of_no_label_train
    # print('num of pathes in training set with labeled end', num_of_has_label_train)
    # print('num of pathes in training set without labeled end', num_of_no_label_train)
    # no_label_X_train = dataset.X_train[no_label]
    # no_label_y_train = dataset.y_train[no_label]
    # has_label = np.invert(no_label)
    # dataset.X_train = dataset.X_train[has_label]
    # dataset.y_train = dataset.y_train[has_label]
    # unique_label, label_counts = np.unique(dataset.y_train[:,-1], return_counts=True)
    # print('label {}\tcounts {}'.format(unique_label, label_counts))
    # label_indices = [np.where(dataset.y_train[:,-1]==label)[0] for label in unique_label]

    no_label = dataset.y_train[:,0] == -2
    num_of_no_label_train = np.sum(no_label)
    num_of_has_label_train = num_train-num_of_no_label_train
    print('num of pathes in training set with labeled start', num_of_has_label_train)
    print('num of pathes in training set without labeled start', num_of_no_label_train)
    no_label_X_train = dataset.X_train[no_label]
    no_label_y_train = dataset.y_train[no_label]
    has_label = np.invert(no_label)
    dataset.X_train = dataset.X_train[has_label]
    dataset.y_train = dataset.y_train[has_label]
    has_label_X_train = dataset.X_train
    has_label_y_train = dataset.y_train
    unique_label, label_counts = np.unique(dataset.y_train[:,0], return_counts=True)
    print('label {}\tcounts {}'.format(unique_label, label_counts))
    # label_indices = [np.where(dataset.y_train[:,-1]==label)[0] for label in unique_label]

    five_path = []
    seven_path = []
    nine_path = []
    for i, path in enumerate(no_label_X_train):
        if len(path) == 5:
            five_path.append(i)
        elif len(path) == 7:
            seven_path.append(i)
        elif len(path) == 9:
            nine_path.append(i)

    five_path_train = []
    seven_path_train = []
    nine_path_train = []
    for i, path in enumerate(dataset.X_train):
        if len(path) == 5:
            five_path_train.append(i)
        elif len(path) == 7:
            seven_path_train.append(i)
        elif len(path) == 9:
            nine_path_train.append(i)

    print('train: 5:{}, 7:{}, 9:{}'.format(len(five_path_train), len(seven_path_train), len(nine_path_train)))
    print('test: 5:{}, 7:{}, 9:{}'.format(len(five_path), len(seven_path), len(nine_path)))

    # embed_kwargs = {
    #     'data_dir': args.data_dir,
    #     'learning_rate': args.learning_rate,
    #     'embed_size': args.embed_size,
    #     'batch_size': args.batch_size_for_embed,
    #     'num_epoch': args.num_epoch_for_embed,
    #     'classifier_hidden_dim': args.classifier_hidden_dim,
    #     'classifier_output_dim': args.classifier_output_dim
    # }

    embed = embedding_loader(args.id_path, args.embed_path, args.embed)

    # embed = nn.Embedding(num_node, args.embed_size)
    # embed.weight.data.copy_(torch.from_numpy(embedding_loader(args.id_path, args.embed_path, args.embed)))

    # embed, classifier = train_embedding(**embed_kwargs)

    # embed = nn.Embedding(num_node, args.embed_size)


    print('Creating embedding...', num_node, args.embed_size)

    # args.alpha = len(dataset.y_train)/len(dataset.y_test)
    # args.beta = 0.01
    # num_real_train = int(args.beta*num_of_has_label_train)
    num_for_weight = 10240
    num_real_train = 10240
    num_for_train = num_for_weight*3
    num_for_test = 1024*1000
    batch_size_for_test = 10240

    args.num_module = dataset.nn_num
    kwargs = {
        'alpha': args.alpha,
        'num_author':dataset.author_num,
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
    # optimizer = torch.optim.Adam([{'params':execution_engine.module_params, 'lr':5e-2},
    #                               {'params':execution_engine.classifier.parameters(), 'lr':5e-4},
    #                                {'params':execution_engine.entity_embeds.parameters(),'lr':5e-1}], lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    # for i in execution_engine.parameters():
    #     print(i)

    # stats = {
    #     'train_losses': [], 'train_losses_ts': [], 'train_losses_ms': [],
    #     'train_accs': [], 'val_accs': [], 'val_accs_es': [],
    #     'best_val_acc': -1, 'model_e': 0,
    # }


    t = 0
    m = 0
    epoch = 0
    # embedding_data = execution_engine.entity_embeds.weight.data

    print('='*80)
    print("Starting training our model...")
    while epoch < args.num_epoch:

        # dataset.shuffle()

        # start_with_label = dataset.y_train[:,0]!=-2
        # start_without_label = np.invert(start_with_label)
        # start_with_label, = np.where(start_with_label)
        # start_without_label, = np.where(start_without_label)
        # start_with_label = np.random.choice(start_with_label, size=num_real_train)
        # start_without_label = np.random.choice(start_without_label, size=num_real_train)
        # random_choice = np.append(start_with_label, start_without_label)
        # random_choice.sort()

        # random_choice = [np.random.choice(indices, size=num_real_train) for indices in label_indices]
        # random_choice = np.concatenate(random_choice)
        # random_choice.sort()


        # random_choice = np.random.choice(range(num_of_has_label_train), size=num_real_train)
        # X_train = dataset.X_train[random_choice]
        # y_train = dataset.y_train[random_choice]

        five_choice = np.random.choice(five_path_train, size=500)
        seven_choice = np.random.choice(seven_path_train, size=1000)
        nine_choice = np.random.choice(nine_path_train, size=3000)
        random_choice = np.concatenate([five_choice,seven_choice,nine_choice])
        np.random.shuffle(random_choice)

        epoch += 1
        print('='*80)
        print('Starting epoch %d' % epoch)
        loss_aver = 0
        num_correct1, num_correct2 = 0, 0
        wrong_has_label, wrong_no_label, right_has_label, right_no_label = 0, 0, 0, 0

        entity2preds1 = dict(zip(dataset.X_test, np.zeros((num_test, args.classifier_output_dim))))
        entity2preds2 = dict(zip(dataset.X_test, np.zeros((num_test, args.classifier_output_dim))))
        entity2preds3 = dict(zip(dataset.X_test, np.zeros((num_test, args.classifier_output_dim))))

        execution_engine.train()

        num_batch = 0
        for batch in dataset.next_batch(dataset.X_train[random_choice], dataset.y_train[random_choice], batch_size=args.batch_size):
            t += 1
            num_batch += 1
            paths, origin_label = batch
            # labels = origin_label[:,-1]
            labels = origin_label[:, 0]
            label_var = Variable(torch.LongTensor(labels).cuda())
            optimizer.zero_grad()
            scores = execution_engine(paths)
            loss = loss_fn(scores, label_var)
            loss_aver += loss.data[0]
            loss.backward()
            optimizer.step()
            # scores = scores.data.cpu().numpy()
            # preds = np.argmax(scores, axis=1)
            # num_correct1 += np.sum(preds == labels)

    # if t % args.record_loss_every == 0:
        loss_aver /= num_batch
        print('=' * 80)
        print(t*args.batch_size, m, 'loss:', loss_aver)
        # stats['train_losses'].append(loss_aver)
        # stats['train_losses_ts'].append(t)
        # stats['train_losses_ms'].append(m)
        loss_aver = 0

        execution_engine.eval()

        print('start evaluate training pathes...')

        # random_choice = np.random.choice(range(num_of_has_label_train), size=num_for_weight)
        # five_choice = np.random.choice(five_path_train, size=num_for_weight)
        # seven_choice = np.random.choice(seven_path_train, size=num_for_weight)
        # nine_choice = np.random.choice(nine_path_train, size=num_for_weight)
        # random_choice = np.concatenate([five_choice, seven_choice, nine_choice])
        seven_choice = np.random.choice(seven_path_train, size=10000)
        nine_choice = np.random.choice(nine_path_train, size=10000)
        random_choice = np.concatenate([five_path_train, seven_choice, nine_choice])
        num_for_train = len(random_choice)
        X_train = has_label_X_train[random_choice]
        y_train = has_label_y_train[random_choice]
        # X_train = dataset.X_train[random_choice]
        # y_train = dataset.y_train[random_choice]
        correct_path = []
        wrong_path = []

        num_correct1, num_correct2 = 0, 0

        for batch in dataset.next_batch(X_train, y_train, batch_size=batch_size_for_test):
            paths, origin_label = batch
            # labels = origin_label[:,-1]
            labels = origin_label[:, 0]
            scores = execution_engine(paths)
            scores = scores.data.cpu().numpy()
            preds = np.argmax(scores, axis=1)
            correct = preds == labels
            num_correct1 += np.sum(correct)
            correct_path += paths[correct].tolist()
            wrong_path += paths[np.invert(correct)].tolist()

        correct_length, correct_leng_count = np.unique([len(i) for i in correct_path], return_counts=True)
        print('correct length and counts', correct_length, correct_leng_count)
        wrong_length, wrong_leng_count = np.unique([len(i) for i in wrong_path], return_counts=True)
        print('wrong length and counts', wrong_length, wrong_leng_count)
        correct_ratio = correct_leng_count / (correct_leng_count+wrong_leng_count)
        random_choice_weight = correct_ratio / np.sum(correct_ratio)
        print('correct rate', correct_ratio, 'choice weight', random_choice_weight)

        print('=' * 80)
        print('start evaluate test pathes...')
        # five_choice, seven_choice, nine_choice = num_for_test * random_choice_weight
        # five_choice = np.random.choice(five_path, size=int(five_choice))
        # seven_choice = np.random.choice(seven_path, size=int(seven_choice))
        # nine_choice = np.random.choice(nine_path, size=int(nine_choice))
        # random_choice = np.concatenate([five_choice,seven_choice,nine_choice])

        # random_choice = np.random.choice(range(num_of_no_label_train), size=num_for_test)
        seven_choice = np.random.choice(seven_path, size=4000)
        nine_choice = np.random.choice(nine_path, size=20000)
        random_choice = np.concatenate([five_path,seven_choice,nine_choice])
        num_for_test = len(random_choice)

        correct_path = []
        wrong_path = []

        for batch in dataset.next_batch(no_label_X_train[random_choice], no_label_y_train[random_choice], batch_size=batch_size_for_test):
            paths, origin_label = batch
            # test = [path[-1] for path in paths]
            test = [path[0] for path in paths]
            labels = np.array([entity2label[i] for i in test])
            scores = F.softmax(execution_engine(paths), dim=1)
            scores = scores.data.cpu().numpy()
            preds = np.argmax(scores, axis=1)
            correct = preds == labels
            num_correct2 += np.sum(correct)

            # wrong = preds != labels
            # start_labels = origin_label[wrong][:, 0]
            # wrong_has_label += np.sum(start_labels!=-2)
            # wrong_no_label += np.sum(start_labels==-2)
            # right = preds == labels
            # start_labels = origin_label[right][:, 0]
            # right_has_label += np.sum(start_labels != -2)
            # right_no_label += np.sum(start_labels == -2)

            correct_path+=paths[correct].tolist()
            wrong_path+=paths[np.invert(correct)].tolist()

            for i in range(len(labels)):
                entity2preds1[test[i]] += np.log(scores[i])
                pred = preds[i]
                if scores[i][pred] > args.threshold:
                    entity2preds2[test[i]][pred] += 1
                entity2preds3[test[i]][pred] += 1

        # print('wrong has label', wrong_has_label)
        # print('wrong no label', wrong_no_label)
        # print('right has label', right_has_label)
        # print('right no label', right_no_label)

        correct_length, correct_leng_count = np.unique([len(i) for i in correct_path], return_counts=True)
        print('correct length and counts', correct_length, correct_leng_count)
        wrong_length, wrong_leng_count = np.unique([len(i) for i in wrong_path], return_counts=True)
        print('wrong length and counts', wrong_length, wrong_leng_count)
        correct_ratio = correct_leng_count / (correct_leng_count + wrong_leng_count)
        print('correct rate', correct_ratio)

        train_acc = float(num_correct1) / num_for_train
        print('train path accuracy is ', train_acc)
        test_acc = float(num_correct2) / num_for_test
        print('test path accuracy is ', test_acc)

        for i, entity2preds in enumerate([entity2preds1,entity2preds2,entity2preds3]):
            en_results = np.array([entity2preds[i] for i in dataset.X_test])
            en_preds = np.argmax(en_results, axis=1)
            en_correct = en_preds == dataset.y_test
            en_acc = float(np.sum(en_correct)) / num_test
            print('ensemble mode {}: acc={}'.format(i, en_acc))
            # max_votes = np.max(en_results, axis=1)
            # votes = np.sum(en_results, axis=1)
            # votes_rate = max_votes/votes
            # np.nan_to_num(votes_rate, copy=False)
            # vrc = votes_rate[en_correct]
            # vrc_mean = np.mean(vrc)
            # vrc_min = np.min(vrc)
            # vrc_max = np.max(vrc)
            # vrw = votes_rate[np.invert(en_correct)]
            # vrw_mean = np.mean(vrw)
            # vrw_min = np.min(vrw)
            # vrw_max = np.max(vrw)
            # print('vote rate for correct mean={} min={} max={}'.format(vrc_mean, vrc_min, vrc_max))
            # print('vote rate for wrong mean={} min={} max={}'.format(vrw_mean, vrw_min, vrw_max))

        execution_engine.train()

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
                  .format(epoch, train_acc, test_acc, en_acc))
        log.write('id\tresults\t\t\t\tlabel\tcorrect\tvote rate\n')
        log_info = np.concatenate((dataset.X_test.reshape(num_test,1), en_results, dataset.y_test.reshape(num_test,1), en_correct.reshape(num_test,1)), axis=1)
        log_info = sorted(log_info, key=lambda x:x[-1])
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

        # embedding = execution_engine.author_embeds.weight.data.cpu().numpy()
        # train_embedding(baseline_dataset, embedding, args)

    print("training is done!")
    # print("best validate accuracy:{}".format(stats['best_val_acc']))


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