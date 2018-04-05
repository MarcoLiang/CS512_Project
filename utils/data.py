import codecs
from collections import OrderedDict
import numpy as np
from itertools import repeat

class Data:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.author_num = None
        self.nn_num = None
        self.bias_num = None
        self.N = None

    #
    # def combine_data(self, datasets):
    #     X_dict = OrderedDict() # author pair -> X
    #     y_dict = OrderedDict() # author pair -> y
    #     curr_len = 1
    #     for dataset in datasets:
    #         with codecs.open(dataset) as f:
    #             for line in f:
    #                 toks = list(map(int, line.strip().split("\t")))
    #                 # print(toks)
    #                 a_pair = (toks[0], toks[-3])
    #
    #                 # print(a_pair)
    #                 label = toks[-1]
    #                 # print(label)
    #                 if not a_pair in X_dict:
    #                     X_dict[a_pair] = [[] for i in repeat(None, len(datasets))]
    #                     y_dict[a_pair] = []
    #                 X_dict[a_pair][curr_len - 1].append(toks[:-1])
    #                 y_dict[a_pair] = label
    #                 self.author_set.add(toks[0])
    #                 self.author_set.add(toks[-3])
    #                 self.pattern_set.add(tuple(toks[:-2]))
    #
    #         curr_len += 1
    #
    #     self.X = np.array(list(X_dict.values()))
    #     self.y = np.array(list(y_dict.values()))
    #     self.author_num = len(self.author_set)
    #     self.pattern_num = len(self.pattern_set)

    def load_data(self, dataset_dir, dataset_cnt_dir):
        print("Loading Data...")
        with codecs.open(dataset_cnt_dir, 'r', 'utf-8') as cntfile:
            for line in cntfile:
                toks = line.strip().split("\t")
                toks = list(map(int, toks))
                self.author_num, self.nn_num, self.bias_num, self.N = toks

        self.X = []
        self.y = np.zeros((int(self.N / 2), 1))

        with codecs.open(dataset_dir, 'r', 'utf-8') as cntfile:
            data_pair = []
            for i, line in enumerate(cntfile):
                toks = line.strip().split("\t")
                toks = list(map(int, toks))
                # x = np.array(toks[:-1])
                x = toks[:-1]
                if i % 2 == 0:
                    data_pair.append(x)
                else:
                    data_pair.append(x)
                    self.X.append(data_pair)
                    data_pair = []
                self.y[int(i / 2)] = toks[-1]
        self.X = np.array(self.X)


    def shuffle(self):
        indices = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

    def split_dataset(self, ratio=[0.7, 0.15, 0.15], shuffle=True):
        '''
        :param ratio(list): the ratio of train, test and valit. e.g. [0.7, 0.15, 0.15]
        :return: three datasets
        '''
        print("Spliting data...")
        n = len(self.y)
        if shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)
        train_idx = indices[0 : int(np.floor(ratio[0] * n))]
        test_idx = indices[int(np.ceil(ratio[0] * n)) : int(np.floor((ratio[0] + ratio[1]) * n))]
        valid_idx = indices[int(np.ceil((ratio[0] + ratio[1]) * n)):]
        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        self.X_valid = self.X[valid_idx]
        self.y_valid = self.y[valid_idx]

    def next_batch(self, X, y, batch_size=1):
        for i in np.arange(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]






#
# data = Data()
# data.load_data("../data/pattern/meta_path_l1_new.txt", "../data/pattern/meta_path_l1_new_cnt.txt")
# print(data.X)
# data.split_dataset()
# print("======")
# print(data.X_train)
# print("======")
# data.shuffle()
# print(data.X_train)





