import codecs

from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn

author_embeddings = dict()
author_id = dict()

# id file has format <id"\t"name>
def load_id_file(id_path, mode='m2v'):
    if mode == 'm2v':
        with codecs.open(id_path,'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if "a" + line[1] in author_id:
                    print("Warning: duplicate author {}".format(line[1]))
                author_id["a" + line[1].replace(' ', '')] = int(line[0])
    elif mode == 'esim':
        pass

def load_pre_trained_emb(emb_path, mode='m2v'):
    word_vectors = KeyedVectors.load_word2vec_format(emb_path, binary=True)
    emb_matrix = np.zeros([len(author_id) + 1, word_vectors.vector_size])
    cnt = 0;
    for i, v in enumerate(word_vectors.index2word):
        if v.startswith("a"):
            if v not in author_id:
                print(v)
                continue
            cnt +=1
            emb_matrix[author_id[v]] = word_vectors.vectors[i]
    return emb_matrix

if __name__ == "__main__":
    id_path = "/Users/ruiyangwang/Desktop/ResearchProject/net_dblp/output/focus/venue_filtered/id_author.txt"
    emb_path = "/Users/ruiyangwang/Desktop/ResearchProject/CS512_Project/embedding_file/m2v/dblp_reduced_focus.cac.w1000.l100"
    load_id_file(id_path)
    m = load_pre_trained_emb(emb_path)
    torch_emb = nn.Embedding(m.shape[0], m.shape[1])
    torch_emb.weight.data.copy_(torch.from_numpy(m))
    print("Done")