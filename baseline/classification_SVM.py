import codecs

from gensim.models import KeyedVectors
from sklearn import multiclass, svm, metrics
import numpy as np
from sklearn.externals import joblib

target_author = dict()
target_author_path = "../data/name-group.txt"  # path to label file
with codecs.open(target_author_path, 'r', 'utf-8') as f:
    for line in f.readlines():
        line = line.strip().split("\t")
        target_author["a" + line[0].replace('_', '')] = line[1]

path = "../data/dblp.cac.w1000.l100"  # path to embedding binary

word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
x_train = []
y_train = []
cnt = 0
for i, v in enumerate(word_vectors.index2word):
    if v.startswith("a"):
        x_train.append(word_vectors.vectors[i])
        y_value = 5
        if v in target_author:
            y_value = target_author[v]
            cnt = cnt + 1
        y_train.append([y_value])

x = np.array(x_train)
y = np.array(y_train)

x_train = x[:int((0.8 * len(x)))]
y_train = y[:int((0.8 * len(x)))]

x_test = x[int(0.8 * len(x)):]
y_test = y[int(0.8 * len(x)):]

print("Training:\n")
model = multiclass.OneVsOneClassifier(svm.LinearSVC())
model.fit(x_train, y_train)
print("Saving model to model.pkl\n")
joblib.dump(model, 'model.pkl')
print("Accuracy on test\n")
y_test_predicted = model.predict(x_test)
metrics.accuracy_score(y_test, y_test_predicted)
print("Accuracy on train\n")
y_train_predicted = model.predict(x_train)
metrics.accuracy_score(y_train, y_train_predicted)
