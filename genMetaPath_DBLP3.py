import sys
import codecs
from utils.global_id import *
from collections import Counter
import itertools
import numpy as np
import random

# Assume the given id is consective and in pattern:
# author id in range [0, author number)
# conference id in range [author number, author number + conf number)
# paper id in range [author number + conf number, total entities number)


class MetaPathGenerator:
    def __init__(self, max_path_length, type_num=3, label_by = 'focus'):
        self.id_author = dict() # author_id(int) -> author_name(str)
        # self.author_id = dict() # author_name(str) -> author_id(int)
        self.a_id_train = None # numpy array
        self.id_conf = dict() # paper_id(int) -> conf_id(int)
        self.id_paper = dict() # paper_id(int) -> paper_title(str)
        self.paper_author = dict() # paper_id(int) -> author_id(int)
        self.author_paper = dict() # author_id(int) -> paper_id(int)
        self.paper_conf = dict() # paper_id(int) -> conf_id(int)
        self.conf_paper = dict() # conf_id(int) -> paper_id(int)
        self.paper_to_paper = dict() # p1 -> p2 means p1 cite p2
        self.paper_from_paper = dict() # p1 <- p2 means p1 was cited by p2
        self.max_path_length = max_path_length # the max entities # in a meta-path
        self.type_num = type_num # max number of entities type in the HIN, default is 3
        self.author_type = 0
        self.conf_type = 1
        self.paper_type = 2
        self.dict_list = [self.id_author, self.id_conf, self.id_paper]
        self.author_group = dict() # author_name(str) -> group_id(str)
        self.author_focus = dict() # author_name(str) -> group_id(str)
        self.nn_list = []
        self.label_by = label_by
        self.N = 0

    def read_data(self, dirpath):
        lst = []
        with codecs.open(dirpath + "/id_author.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[int(toks[0])] = toks[1]
                    # self.author_id[toks[1]] = int(toks[0])
                    lst.append(int(toks[0]))
        self.a_id_train = set(random.sample(lst, k=len(self.id_author) // 3))

        with codecs.open(dirpath + "/paper.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_paper[int(toks[0])] = toks[1]

        with codecs.open(dirpath + "/id_conf.txt", 'r', 'utf-8') as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    newconf = toks[1]
                    self.id_conf[int(toks[0])] = newconf

        with codecs.open(dirpath + "/paper_author.txt", 'r', 'utf-8') as pafile:
            for line in pafile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = int(toks[0]), int(toks[1])
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
                    p, c = int(toks[0]), int(toks[1])
                    self.paper_conf[p] = c
                    if c not in self.conf_paper:
                        self.conf_paper[c] = []
                    self.conf_paper[c].append(p)

        with codecs.open(dirpath + "/paper_paper.txt", 'r', 'utf-8') as ppfile:
            for line in ppfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p1, p2 = int(toks[0]), int(toks[1]) # p1 cited p2, p1 -> p2
                    if p1 not in self.paper_to_paper:
                        self.paper_to_paper[p1] = set() # p1 cited p2
                    self.paper_to_paper[p1].add(p2)
                    if p2 not in self.paper_from_paper:
                        self.paper_from_paper[p2] = set()
                    self.paper_from_paper[p2].add(p1)

        if self.label_by == 'group':
            with codecs.open("data/name-group.txt", 'r', 'utf-8') as agfile:
                for line in agfile:
                    toks = line.strip().split("\t")
                    toks[0] = toks[0].replace('_', ' ')
                    self.author_group[toks[0]] = int(toks[1])
        elif self.label_by == 'focus':
            with codecs.open("data/name-focus.txt", 'r', 'utf-8') as affile:
                for line in affile:
                    toks = line.strip().split("\t")
                    toks[0] = toks[0].replace('_', ' ')
                    self.author_focus[toks[0]] = int(toks[1])

    def get_entity_type(self, entity_id):
        author_num = len(self.id_author)
        conf_num = len(self.id_conf)
        # print(entity_id)
        # print(author_num)
        # print(conf_num)
        if entity_id < author_num:
            # print('t{}'.format(self.author_type))
            return self.author_type

        elif entity_id < author_num + conf_num:
            # print('t{}'.format(self.conf_type))
            return self.conf_type
        else:
            # print('t{}'.format(self.paper_type))
            return self.paper_type

    def write_pattern_path(self, meta_path, file, inTest):
        '''
        authro_id_out = author_id(local) - 1
        paper_id_out = paper_id(local) - 1
        conf_id_out = conf_id(local) - 1 + bias_offset


        [paper1 = 3, paper2 = 2] means paper1 cite paper2
        [paper1 = 2, paper2 = 3] means paper1 was cited by paper2

        '''
        self.N += 1
        # [author0, author1,...,author_k = offset - 1, paper0,...,paper_k=offset1+offset2-1, conf0 = offset1+offset2]
        # entities list
        entities_fwd = []
        for e in meta_path:
            entities_fwd.append(e)

        # edges (nn) list
        edges_fwd = []
        for pair_id in zip(meta_path, meta_path[1:]):
            pair_type = self.meta_path_type(pair_id)
            if pair_type == [self.paper_type, self.paper_type]:
                if pair_id[0] in self.paper_to_paper and pair_id[1] in self.paper_to_paper[pair_id[0]]:
                    pair_type[0] = 3
                else:
                    pair_type[1] = 3
            if not pair_type in self.nn_list:
                self.nn_list.append(pair_type)
            edges_fwd.append(self.nn_list.index(pair_type))

        path_fwd = [0] * (len(entities_fwd) + len(edges_fwd))

        path_fwd[::2] = entities_fwd
        path_fwd[1::2] = edges_fwd
        if self.label_by == 'focus':
            path_fwd.append(self.author_focus[self.id_author[meta_path[0]]])
            path_fwd.append(self.author_focus[self.id_author[meta_path[-1]]])
        elif self.label_by == 'group':
            path_fwd.append(self.author_group[self.id_author[meta_path[0]]])
            path_fwd.append(self.author_group[self.id_author[meta_path[-1]]])
        if inTest:
            path_fwd[-1] = -1
        path_fwd_str = '\t'.join(list(map(str, path_fwd)))
        print(path_fwd_str + '\t')
        file.write(path_fwd_str + '\n')

    def meta_path_type(self, meta_path):
        return [self.get_entity_type(id) for id in meta_path]


    def generate_metapath(self, dir_out_train, dir_out_test):
        marked_author = set() # store the id of marked author
        DBLP_train = open(dir_out_train + '/DBLP_train.txt', 'w')
        for author_id in self.a_id_train:
            stack = Stack()
            marked_author.add(author_id)
            meta_path = []
            stack.push(Node(author_id, 0, 0))

            while not stack.isEmpty():
                node = stack.pop()
                entity_type = self.get_entity_type(node.id)
                # add entity to current meta-path
                if node.step == len(meta_path):
                    meta_path.append(node.id)
                else:
                    meta_path[node.step] = node.id
                # check terminate criteria
                if entity_type == self.author_type and (not node.id in marked_author):
                    inTestSet = node.id in self.a_id_train
                    self.write_pattern_path(meta_path[:node.step + 1], DBLP_train, inTestSet)
                    continue
                if node.step == self.max_path_length - 1:
                    continue
                # continue traverse the graph and generate neighbors
                if node.step == 0: # generate neighbors form the start author
                    for p_id in self.author_paper[node.id]:
                        stack.push(Node(p_id, node.step + 1, 1))
                elif entity_type == self.conf_type:
                    for p_id in self.conf_paper[node.id]:
                        if not node.id in meta_path[:node.step]: # check circle
                            stack.push(Node(p_id, node.step + 1, 1))
                elif entity_type == self.paper_type:
                    # paper to paper relation, stop if more than 3 consective papers
                    if node.paper_num < 3:
                        if node.id in self.paper_to_paper:
                            for p_id in self.paper_to_paper[node.id]:
                                if not node.id in meta_path[: node.step]:
                                    stack.push(Node(p_id, node.step + 1, node.paper_num + 1))
                        if node.id in self.paper_from_paper: # some papers are not be cited
                            for p_id2 in self.paper_from_paper[node.id]:
                                if not node.id in meta_path[: node.step]:
                                    stack.push(Node(p_id2, node.step + 1, node.paper_num + 1))
                    # paper conference relation
                    if node.id in self.paper_conf:
                        conf_id = self.paper_conf[node.id]
                        if not node.id in meta_path[:node.step]:
                            stack.push(Node(conf_id, node.step + 1, 0))
                    # paper author relation
                    for a_id in self.paper_author[node.id]:
                        if not node.id in meta_path[:node.step]:
                            stack.push(Node(a_id, node.step + 1, 0))
        DBLP_train.close()
        out_file_stat = open(dir_out_train + '/meta_path_l1_new_cnt.txt', 'w')
        header = [len(self.id_author), len(self.nn_list), len(self.id_paper) + len(self.id_conf), self.N]
        out_file_stat.write('\t'.join(list(map(str, header))))

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
    def __init__(self, entity_id, step, paper_num=0):
        self.id = entity_id
        self.step = step
        self.paper_num = paper_num



def main():
    dir_input = "data/focus/venue_filtered_unique_id"
    dir_output = "data/pattern"
    meta = MetaPathGenerator(5, 3, "focus")
    meta.read_data(dir_input)
    # print(meta.author_focus)
    print('Generating Meta-Path using files in {0}, label by {1}'.format(dir_input, meta.label_by))
    meta.generate_metapath(dir_output, dir_output)
    print('====================================================')
    print('Mata-Path (training set) stored in {}: meta_path_DBLP.txt: '.format('data/classify_task'))
    print('num_of_author, num_of_edges(nn), num_of_bias stored in {}: meta_path_l1_new_cnt.txt'.format(dir_output) )

if __name__ == "__main__":
    main()