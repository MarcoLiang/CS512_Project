import sys
import codecs
from utils.global_id import *
from collections import Counter



class MetaPathGenerator:
    def __init__(self, path_length, type_num):
        self.id_author = dict()
        self.id_conf = dict()
        self.id_paper = dict()
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()
        self.paper_to_paper = dict() # p1 -> p2 means p1 cited p2
        self.paper_from_paper = dict()
        self.path_length = path_length
        self.type_num = type_num
        self.author_type = 0
        self.conf_type = 1
        self.paper_type = 2
        self.max_path_length = 5
        self.dict_list = [self.id_author, self.id_conf, self.id_paper]
        self.author_group = dict()

    def read_data(self, dirpath):
        with codecs.open(dirpath + "/id_author.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[toks[0]] = toks[1]

        with codecs.open(dirpath + "/id_conf.txt", 'r', 'utf-8') as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    newconf = toks[1]
                    self.id_conf[toks[0]] = newconf

        with codecs.open(dirpath + "/paper_author.txt", 'r', 'utf-8') as pafile:
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

        with codecs.open(dirpath + "/paper_conf.txt", 'r', 'utf-8') as pcfile:
            for line in pcfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, c = toks[0], toks[1]
                    self.paper_conf[p] = c
                    if c not in self.conf_paper:
                        self.conf_paper[c] = []
                    self.conf_paper[c].append(p)

        with codecs.open(dirpath + "/paper_paper.txt") as ppfile:
            for line in ppfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p1, p2 = toks[0], toks[1] # p1 cited p2, p1 -> p2
                    # print(p1)
                    if p1 not in self.paper_to_paper:
                        self.paper_to_paper[p1] = [] # p1 cited p2
                    self.paper_to_paper[p1].append(p2)
                    if p2 not in self.paper_from_paper:
                        self.paper_from_paper[p2] = []
                    self.paper_from_paper[p2].append(p1)

        with codecs.open("data/name-group.txt") as agfile:
            for line in agfile:
                toks = line.strip().split("\t")
                toks[0] = toks[0].replace('_', ' ')
                self.author_group[toks[0]] = toks[1]

    # def write_file_id(self, meta_path, file):
    #     '''
    #     write global id
    #     '''
    #     meta_str = '\t'.join(list(map(str, meta_path))) + '\n'
    #     file.write(meta_str)

    # def write_file_str(self, meta_path, file):
    #     meta_str = [self.dict_list[retrieve_type(g_id)][str(retrieve_id(g_id))] for g_id in meta_path]
    #     meta_str = '\t'.join(meta_str) + '\n'
    #     file.write(meta_str)

    # def write_file_pattern1(self, pattern_counter, file_pos, file_neg):
    #     print(pattern_counter)
    #     for k, v in pattern_counter.items():
    #         out_str = [str(i) for i in k]
    #         out_str = '\t'.join(out_str)
    #         file_pos.write(out_str + '\t' + str(v[0]) + '\n')
    #         file_neg.write(out_str + '\t' + str(v[1]) + '\n')

    def write_file_pattern_full(self, meta_path, file):
        # meta_path = [retrieve_id(e) for e in meta_path]
        meta_str = '\t'.join(list(map(str, meta_path)))
        # file.write(str(meta_path[0]) + '\t')
        file.write(meta_str + '\n')
        # file.write(str(meta_path[-1]) + '\t')
        # file.write(str(self.check_same_group(meta_path)) + '\n')


    def meta_path_type(self, meta_path):
        return [retrieve_type(id) for id in meta_path]

    def check_same_group(self, meta_path):
        id1 = self.id_author[str(retrieve_id(meta_path[0]))]
        id2 = self.id_author[str(retrieve_id(meta_path[-1]))]
        return int(self.author_group[id1] == self.author_group[id2])

    def generate_metapath(self, dir_out):
        marked_author = set() # store the global_id of marked author
        out_file_pattern_full = open(dir_out + '/meta_path_pattern_l1_all.txt', 'w')
        pattern_counter = {}
        for author_id in self.id_author.keys():
            stack = Stack()
            marked_author.add(author_id)
            meta_path = []
            stack.push(Node(author_id, self.author_type, 0))
            while not stack.isEmpty():
                node = stack.pop()
                # add entity to current meta-path (global id)
                if node.step == len(meta_path):
                    meta_path.append(get_global_id(node.id, node.type))
                else:
                    meta_path[node.step] = get_global_id(node.id, node.type)

                if len(meta_path) > self.max_path_length:
                    print('xxxxx')

                # check terminate criteria
                if node.type == self.author_type and (not node.id in marked_author):
                    # ============== write all patterns ===================#
                    meta_path_type = tuple(self.meta_path_type(meta_path[1:node.step]))
                    if not meta_path_type in pattern_counter:
                        pattern_counter[meta_path_type] = [0, 0]
                    elif self.check_same_group(meta_path[:node.step + 1]):
                        pattern_counter[meta_path_type][0] += 1
                    else:
                        pattern_counter[meta_path_type][1] += 1
                    # ============== write patterns without authors ========#
                    self.write_file_pattern_full(meta_path[:node.step + 1], out_file_pattern_full)
                    # ======================================================#
                    continue
                if node.step == self.max_path_length - 1:
                    continue
                # continue traverse the graph and generate neighbors
                if node.step == 0: # generate neighbors form the start author
                    # print('node nodeid:{0} type:{1} step:{2}'.format(node.id, node.type, node.step))
                    for p_id in self.author_paper[node.id]:
                        stack.push(Node(p_id, self.paper_type, node.step + 1, 1))
                if node.type == self.conf_type:
                    for p_id in self.conf_paper[node.id]:
                        if not get_global_id(p_id, self.paper_type) in meta_path[:node.step]: # check circle
                            stack.push(Node(p_id, self.paper_type, node.step + 1, 1))
                if node.type == self.paper_type:
                    # paper to paper relation, stop if more than 3 consective papers
                    if node.paper_num < 3:
                        if node.id in self.paper_to_paper:
                            for p_id in self.paper_to_paper[node.id]:
                                if not get_global_id(p_id, self.paper_type) in meta_path[: node.step]:
                                    stack.push(Node(p_id, self.paper_type, node.step + 1, node.paper_num + 1))
                        if node.id in self.paper_from_paper: # some papers are not be cited
                            for p_id2 in self.paper_from_paper[node.id]:
                                if not get_global_id(p_id2, self.paper_type) in meta_path[: node.step]:
                                    stack.push(Node(p_id2, self.paper_type, node.step + 1, node.paper_num + 1))
                    # paper conference relation
                    if node.id in self.paper_conf:
                        conf_id = self.paper_conf[node.id]
                        if not get_global_id(conf_id, self.conf_type) in meta_path[:node.step]:
                            stack.push(Node(conf_id, self.conf_type, node.step + 1))
                    # paper author relation
                    for a_id in self.paper_author[node.id]:
                        if not get_global_id(a_id, self.author_type) in meta_path[:node.step]:
                            stack.push(Node(a_id, self.author_type, node.step + 1))

class Stack:
    def __init__(self):
        self.storage = []

    def isEmpty(self):
        return len(self.storage) == 0

    def push(self,p):
        self.storage.append(p)

    def pop(self):
        return self.storage.pop()

    def size(self):
        return len(self.storage)


class Node:
    def __init__(self, entity_id, entity_type, step, paper_num=0):
        self.id = entity_id
        self.type = entity_type
        self.step = step
        self.paper_num = paper_num

    def __hash__(self):
        get_global_id(self.id, self.type)


def main():
    meta = MetaPathGenerator(10, 3)
    meta.read_data("_reduced_dataset/output")
    meta.generate_metapath("_reduced_dataset/pattern")


if __name__ == "__main__":
    main()