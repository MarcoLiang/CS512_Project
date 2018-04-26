import codecs
from collections import OrderedDict
import numpy as np
from itertools import repeat

class BaselineData:
    def __init__(self, dir, split_ratio=[0.8, 0.2], shuffle=True):
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

        self.load_data(dir)
        # self.split_dataset(split_ratio, shuffle)

    def load_data(self, dataset_dir):
        print("Loading Data (baseline)...")
        with open(dataset_dir + '/DBLP_stat.txt') as f:
            for line in f:
                toks = line.strip().split("\t")
                toks = list(map(int, toks))
                self.author_num, self.nn_num, self.bias_num, self.N = toks


        with codecs.open(dataset_dir + '/DBLP_train_baseline.txt', 'r', 'utf-8') as trainset:
            X = []
            y = []
            for line in trainset:
                toks = line.strip().split("\t")
                X.append(int(toks[0]))
                y.append(int(toks[-1]))

            self.X_train = np.array(X)
            self.y_train = np.array(y) - 1

        with codecs.open(dataset_dir + '/DBLP_test.txt', 'r', 'utf-8') as testset:
            X = []
            y = []
            for line in testset:
                toks = line.strip().split("\t")
                X.append(int(toks[0]))
                y.append(int(toks[-1]))

            self.X_test = np.array(X)
            self.y_test = np.array(y)-1

    def shuffle(self):
        indices = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]


    def next_batch(self, X, y, batch_size=1):
        for i in np.arange(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]


#
# dir = "../data/classify_task/pattern"
# data = Data(dir)
# print(data.X_train)
# print(data.y_train)

# print(data.y.shape)
# print(data.X_train.shape)
# print(data.X_valid.shape)
# print(data.X_test.shape)
# print("======")
# print(data.X_train)
# print("======")
# data.shuffle()
# print(data.X_train)





