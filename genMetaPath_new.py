import sys
import codecs
from utils.global_id import *
from collections import Counter
import itertools



class MetaPathGenerator:
    def __init__(self, max_path_length, type_num=3, label_by = 'focus'):
        self.id_author = dict() # author_id(int) -> author_name(str)
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
        self.dict_list = [self.id_author, self.id_conf, self.id_paper, self.id_paper]
        self.author_group = dict() # author_name(str) -> group_id(str)
        self.author_focus = dict()
        self.nn_list = []
        self.label_by = label_by
        self.N = 0

    def read_data(self, dirpath):
        with codecs.open(dirpath + "/id_author.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[int(toks[0])] = toks[1]

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
        else:
            with codecs.open("data/name-focus.txt", 'r', 'utf-8') as affile:
                for line in affile:
                    toks = line.strip().split("\t")
                    toks[0] = toks[0].replace('_', ' ')
                    self.author_focus[toks[0]] = int(toks[1])


    def write_file_pattern_full(self, meta_path, file):
        '''
        authro_id_out = author_id(local) - 1
        paper_id_out = paper_id(local) - 1
        conf_id_out = conf_id(local) - 1 + bias_offset


        [paper1 = 3, paper2 = 2] means paper1 cite paper2
        [paper1 = 2, paper2 = 3] means paper1 was cited by paper2

        '''
        self.N += 2
        bias_offset = len(self.id_paper) # [paper0, paper1,...,paper_k = offset - 1, conf0 = offset]
        # entities list
        entities_fwd = []
        for e in meta_path:
            out_id = int(retrieve_id(e)) - 1
            if retrieve_type(e) == self.conf_type:
                out_id += bias_offset
            entities_fwd.append(out_id)
        entities_bwd = entities_fwd[::-1]

        # edges (nn) list
        edges_fwd = []
        edges_bwd = []
        for pair in zip(meta_path, meta_path[1:]):
            pair_type = self.meta_path_type(pair)
            pair_id = [retrieve_id(i) for i in pair]
            if pair_type == [self.paper_type, self.paper_type]:
                if pair_id[0] in self.paper_to_paper and pair_id[1] in self.paper_to_paper[pair_id[0]]:
                    pair_type[0] = 3
                else:
                    pair_type[1] = 3
            pair_type_reverse = pair_type[::-1]
            if not pair_type in self.nn_list:
                self.nn_list.append(pair_type)
            if not pair_type_reverse in self.nn_list:
                self.nn_list.append(pair_type_reverse)
            edges_fwd.append(self.nn_list.index(pair_type))
            edges_bwd.append(self.nn_list.index(pair_type_reverse))

        edges_bwd = edges_bwd[::-1]
        path_fwd = [0] * (len(entities_fwd) + len(edges_fwd))
        path_bwd = [0] * (len(entities_bwd) + len(edges_bwd))

        label =  self.check_label(meta_path)
        path_fwd[::2] = entities_fwd
        path_fwd[1::2] = edges_fwd
        path_fwd.append(label)
        path_bwd[::2] = entities_bwd
        path_bwd[1::2] = edges_bwd
        path_bwd.append(label)

        path_fwd_str = '\t'.join(list(map(str, path_fwd)))
        path_bwd_str = '\t'.join(list(map(str, path_bwd)))

        file.write(path_fwd_str + '\n')
        file.write(path_bwd_str + '\n')


    def meta_path_type(self, meta_path):
        return [retrieve_type(id) for id in meta_path]

    def check_label(self, meta_path):
        a1 = self.id_author[retrieve_id(meta_path[0])]
        a2 = self.id_author[retrieve_id(meta_path[-1])]
        if self.label_by == 'group':
           label = int(self.author_group[a1] == self.author_group[a2])
        else:
            label = int(self.author_focus[a1] == self.author_focus[a2])
        return label

    def generate_metapath(self, dir_out):
        marked_author = set() # store the global_id of marked author
        out_file_pattern_full = open(dir_out + '/meta_path_l1_new.txt', 'w')
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
                # check terminate criteria
                if node.type == self.author_type and (not node.id in marked_author):
                    # ============== write full path ========#
                    self.write_file_pattern_full(meta_path[:node.step + 1], out_file_pattern_full)
                    # ======================================================#
                    continue
                if node.step == self.max_path_length - 1:
                    continue
                # continue traverse the graph and generate neighbors
                if node.step == 0: # generate neighbors form the start author
                    for p_id in self.author_paper[node.id]:
                        stack.push(Node(p_id, self.paper_type, node.step + 1, 1))
                elif node.type == self.conf_type:
                    for p_id in self.conf_paper[node.id]:
                        if not get_global_id(p_id, self.paper_type) in meta_path[:node.step]: # check circle
                            stack.push(Node(p_id, self.paper_type, node.step + 1, 1))
                elif node.type == self.paper_type:
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
        out_file_pattern_full.close()
        out_file_stat = open(dir_out + '/meta_path_l1_new_cnt.txt', 'w')
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
    def __init__(self, entity_id, entity_type, step, paper_num=0):
        self.id = entity_id
        self.type = entity_type
        self.step = step
        self.paper_num = paper_num

    def __hash__(self):
        get_global_id(self.id, self.type)


def main():
    dir_input = "data/focus/venue_filtered"
    dir_output = "data/pattern"
    meta = MetaPathGenerator(5, 3, "focus")
    meta.read_data(dir_input)
    print('Generating Meta-Path using files in {0}, label by {1}'.format(dir_input, meta.label_by))
    meta.generate_metapath(dir_output)
    print('====================================================')
    print('Mata-Path stored in {}: meta_path_l1_new.txt: '.format(dir_output))
    print('num_of_author, num_of_edges(nn), num_of_bias stored in {}: meta_path_l1_new_cnt.txt'.format(dir_output) )

if __name__ == "__main__":
    main()