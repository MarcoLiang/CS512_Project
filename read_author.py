import codecs
import numpy as np

file = './data/classify_task_170W/pattern_30_70/DBLP_train.txt'
paths = []
labels = []

with codecs.open(file, 'r', 'utf-8') as cntfile:
    for i, line in enumerate(cntfile):
        toks = line.strip().split("\t")
        toks = list(map(int, toks))
        paths.append(toks[:-2])
        labels.append(toks[-2:])

paths = np.array(paths)
labels = np.array(labels) - 1

while True:
    print('input id:')
    id = input()
    print('id is', id)
    for i, path in enumerate(paths):
        if path[-1] == int(id) or path[0] == int(id):
            print(path, labels[i])