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
                    a_pair = (toks[0], toks[-2])
                    label = toks[-1]
                    # print(label)
                    if not a_pair in X_dict:
                        X_dict[a_pair] = []
                        y_dict[a_pair] = []
                    X_dict[a_pair].append(toks[:-1])
                    y_dict[a_pair].append(label)
        self.X = list(X_dict.values())
        self.y = list(y_dict.values())
        return list(X_dict.values()), list(y_dict.values())

    def split_dataset(self, ratio=[0.7, 0.15, 0.15]):
        '''
        :param ratio(list): the ratio of train, test and valit. e.g. [0.7, 0.15, 0.15]
        :return: three datasets
        '''
        N = self.X.shape[0]
        indices = np.random.permutation(N)
        train_idx = indices[0 : np.floor(ratio[0] * N)]
        test_idx = indices[np.ceil(ratio[0] * N) : np.floor((ratio[0] + ratio[1]) * N)]
        valid_idx = indices[np.ceil((ratio[0] + ratio[1]) * N)]
        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        self.X_valid = self.X[valid_idx]
        self.y_valid = self.y[valid_idx]

    def next_batch(self, X, y, batch_size):
        for i in np.arange(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]


meta = Data()
X, y = meta.combine_data(["../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1",
                           "../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2",
                           "../_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l3"])
print(X)
print(y)





