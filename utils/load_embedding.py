import codecs
from collections import Counter

from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn


# id file has format <id"\t"name>
def load_id_file(file_path, mode='m2v'):
    author_id = dict()
    if mode == 'm2v':
        with codecs.open(file_path + '/id_author.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if "a".replace(' ', '') + line[1] in author_id:
                    print("Warning: duplicate author {}".format(line[1]))
                author_id["a" + line[1].replace(' ', '')] = int(line[0])
        f.close()
        with codecs.open(file_path + '/id_conf.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if line[1] in author_id:
                    print("Warning: duplicate conf {}".format(line[1]))
                author_id["v" + line[1].replace(' ', '')] = int(line[0])
        f.close()
    elif mode == 'esim':
        with codecs.open(file_path + '/id_author.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if line[1] in author_id:
                    print("Warning: duplicate author {}".format(line[1]))
                author_id[line[1].replace(' ', '_')] = int(line[0])
        f.close()
        with codecs.open(file_path + '/id_conf.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if line[1] in author_id:
                    print("Warning: duplicate conf {}".format(line[1]))
                author_id[line[1].replace(' ', '_')] = int(line[0])
        f.close()
        with codecs.open(file_path + '/paper.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if line[1] in author_id:
                    print("Warning: duplicate paper {}".format(line[1]))
                author_id[line[1].replace(' ', '_')] = int(line[0])
        f.close()
    return author_id


def embedding_loader(id_path, emb_path, mode='m2v'):
    emb_matrix = None
    if mode == 'm2v':
        word_vectors = KeyedVectors.load_word2vec_format(emb_path, binary=True)
        author_id = load_id_file(id_path, mode)
        emb_matrix = np.zeros([len(author_id), word_vectors.vector_size])
        for i, v in enumerate(word_vectors.index2word):
            if v not in author_id:
                print(v)
                continue
            # cnt += 1
            emb_matrix[author_id[v]] = word_vectors.vectors[i]
    elif mode == 'esim':
        word_vectors = KeyedVectors.load_word2vec_format(emb_path, binary=True)
        author_id = load_id_file(id_path, mode)
        emb_matrix = np.zeros([len(author_id), word_vectors.vector_size])
        cnt = 0
        for i, v in enumerate(word_vectors.index2word):
            if v not in author_id:
                print(v)
                continue
            cnt += 1
            emb_matrix[author_id[v]] = word_vectors.vectors[i]
    elif mode == 'dw':
        word_vectors = KeyedVectors.load_word2vec_format(emb_path, binary=False)
        emb_matrix = np.zeros([len(word_vectors.index2word), word_vectors.vector_size])
        for i, v in enumerate(word_vectors.index2word):
            emb_matrix[int(v)] = word_vectors.vectors[i]
    return emb_matrix


if __name__ == "__main__":
    id_path = "/Users/ruiyangwang/Desktop/ResearchProject/CS512_Project/data/focus/venue_filtered_unique_id"
    emb_path = "/Users/ruiyangwang/Desktop/ResearchProject/CS512_Project/embedding_file/esim/vec_128_new.dat"
    m2vpath = "/Users/ruiyangwang/Desktop/ResearchProject/code_metapath2vec/m2vembedding"
    m = embedding_loader(id_path, m2vpath, "m2v")
    torch_emb = nn.Embedding(m.shape[0], m.shape[1])
    torch_emb.weight.data.copy_(torch.from_numpy(m))
# print("Done")
