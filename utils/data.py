import codecs
from collections import OrderedDict
import numpy as np

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
        self.author_group = dict()


    def combine_data(self, datasets):
        X_dict = OrderedDict() # author pair -> X
        y_dict = OrderedDict() # author pair -> y
        for dataset in datasets:
            with codecs.open(dataset) as f:
                for line in f:
                    toks = list(map(int, line.strip().split("\t")))
                    a_pair = (toks[0], toks[-3])
                    # print(a_pair)
                    label = toks[-1]
                    # print(label)
                    if not a_pair in X_dict:
                        X_dict[a_pair] = []
                        y_dict[a_pair] = []
                    X_dict[a_pair].append(toks[:-1])
                    y_dict[a_pair].append(label)
        # self.X = np.fromiter(X_dict.values(), dtype=int)
        # self.y = np.fromiter(y_dict.values(), dtype=int)
        # print(X_dict)
        self.X = np.array(list(X_dict.values()))
        self.y = np.array(list(y_dict.values()))
        return self.X, self.y.shape

    # def build_dict(self, dataset, X_dict, y_dict):
    #     with codecs.open(dataset) as f:
    #         for line in f:
    #             toks = list(map(int, line.strip().split("\t")))
    #             a_pair = (toks[0], toks[-2])
    #             label = toks[-1]
    #             # print(label)
    #             if not a_pair in X_dict:
    #                 X_dict[a_pair] = []
    #                 y_dict[a_pair] = []
    #             X_dict[a_pair].append(toks[:-1])
    #             y_dict[a_pair].append(label)
    #     return np.array(list(X_dict.values())), np.array(list(y_dict.values()))

    def split_dataset(self, ratio=[0.7, 0.15, 0.15], shuffle=True):
        '''
        :param ratio(list): the ratio of train, test and valit. e.g. [0.7, 0.15, 0.15]
        :return: three datasets
        '''
        N = self.X.shape[0]
        if shuffle:
            indices = np.random.permutation(N)
        else:
            indices = np.arange(N)
        # print(type(indices))
        train_idx = indices[0 : int(np.floor(ratio[0] * N))]
        test_idx = indices[int(np.ceil(ratio[0] * N)) : int(np.floor((ratio[0] + ratio[1]) * N))]
        valid_idx = indices[int(np.ceil((ratio[0] + ratio[1]) * N))]
        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        self.X_valid = self.X[valid_idx]
        self.y_valid = self.y[valid_idx]

    def next_batch(self, X, y, batch_size=1):
        for i in np.arange(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]


    def data_load(self, dir, length, shuffle = True, split_ratio = [0.7, 0.15, 0.15]):
        dataset = dir + "/meta-path_pattern_l"
        datasets = [dataset + str(i) for i in range(1, length + 1)]
        self.combine_data(datasets)
        self.split_dataset(split_ratio, shuffle)




data = Data()
# X, y = meta.combine_data(["../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1",
#                            "../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2",
#                            "../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l3"])



dir = "../_reduced_dataset/filter_venue_since_2005/pattern"
data.data_load(dir, 3)
print(data.X_train[5])
# for X_batch, y_batch in data.next_batch(data.X_train, data.y_train):
#     print(X_batch)
#     print(y_batch)




# print(type(X[0]))
# print(y)





