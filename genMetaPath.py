import sys
import codecs


class MetaPathGenerator:
    def __init__(self, path_length, type_num):
        self.id_author = dict()
        self.id_conf = dict()
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()
        self.paper_to_paper = dict() # p1 -> p2 means p1 cited p2
        self.paper_from_paper = dict()
        self.path_length = path_length
        self.type_num = type_num

    def read_data(self, dirpath):
        with codecs.open(dirpath + "/id_author_1000.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[toks[0]] = toks[1].replace(" ", "")
                    # print(self.id_author[toks[0]])

        # print "#authors", len(self.id_author)

        with codecs.open(dirpath + "/id_conf_1000.txt", 'r', 'utf-8') as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    newconf = toks[1].replace(" ", "")
                    self.id_conf[toks[0]] = newconf
                    # print(newconf)
        #
        # # print "#conf", len(self.id_conf)
        #
        with codecs.open(dirpath + "/paper_author_1000.txt", 'r', 'utf-8') as pafile:
            for line in pafile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_author:
                        self.paper_author[p] = []
                    self.paper_author[p].append(a)
                    if a not in self.author_paper:
                        self.author_paper[a] = []
                    self.author_paper[a].append(p)

        with codecs.open(dirpath + "/paper_conf_1000.txt") as pcfile:
            for line in pcfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, c = toks[0], toks[1]
                    self.paper_conf[p] = c
                    if c not in self.conf_paper:
                        self.conf_paper[c] = []
                    self.conf_paper[c].append(p)

        with codecs.open(dirpath + "/paper_paper_1000.txt") as ppfile:
            for line in ppfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p1, p2 = toks[0], toks[1] # p1 cited p2, p1 -> p2
                    if p1 not in self.paper_to_paper:
                        self.paper_to_paper[p1] = [] # p1 cited p2
                    self.paper_to_paper[p1].append(p2)
                    if p2 not in self.paper_from_paper:
                        self.paper_from_paper[p2]= []
                    self.paper_from_paper[p2] = p1



def main():
    meta = MetaPathGenerator(10, 3)
    meta.read_data("data/dblp_1000")





if __name__ == "__main__":
    main()