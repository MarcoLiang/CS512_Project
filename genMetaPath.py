import sys
import codecs
from utils.global_id import *



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

    def read_data(self, dirpath):
        with codecs.open(dirpath + "/id_author.txt", 'r', 'utf-8') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[toks[0]] = toks[1]#.replace(" ", "")
                    # print(self.id_author[toks[0]])

        # print "#authors", len(self.id_author)

        with codecs.open(dirpath + "/id_conf.txt", 'r', 'utf-8') as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    newconf = toks[1]#.replace(" ", "")
                    self.id_conf[toks[0]] = newconf
                    # print(newconf)
        #
        # # print "#conf", len(self.id_conf)
        #
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


    def write_file_id(self, meta_path, file):
        '''
        write global id
        '''
        meta_str = '\t'.join(list(map(str, meta_path))) + '\n'
        file.write(meta_str)

    def write_file_str(self, meta_path, file):
        meta_str = [self.dict_list[retrieve_type(g_id)][str(retrieve_id(g_id))] for g_id in meta_path]
        meta_str = '\t'.join(meta_str) + '\n'
        file.write(meta_str)

    def print_meta_path_type(self, meta_path):
        meta_path_type = [retrieve_type(id) for id in meta_path]
        print(meta_path_type)

    def generate_metapath(self, dir_out):
        marked_author = set() # store the global_id of marked author
        out_file_id = open(dir_out + '/meta_path_id.txt', 'w')
        # out_file_str = open(dir_out + '/meta_path_str.txt', 'w')
        for author_id in self.id_author.keys():
            print(author_id)
            stack = Stack()
            marked_author.add(author_id)
            meta_path = []
            stack.push(Node(author_id, self.author_type, 0))
            while not stack.isEmpty():
                # print(meta_path)
                node = stack.pop()
                # self.print_meta_path_type(meta_path[:node.step + 1])
                # print('node nodeid:{0} type:{1} step:{2}'.format(node.id, node.type, node.step))
                if node.step == len(meta_path):
                    meta_path.append(get_global_id(node.id, node.type))
                else:
                    meta_path[node.step] = get_global_id(node.id, node.type)

                if len(meta_path) > self.max_path_length:
                    print('xxxxx')

                # check terminate criteria
                if node.type == self.author_type and (not node.id in marked_author):
                    # print('find in')
                    self.write_file_id(meta_path[:node.step + 1], out_file_id)
                    # self.write_file_str(meta_path[:node.step + 1], out_file_str)
                    # self.print_meta_path_type(meta_path[:node.step + 1])
                    # print(meta_path[0:node.step + 1])
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
                            # print(node.id)
                            # print(self.paper_from_paper[node.id])
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
        out_file_id.close()


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
    # print(meta.paper_from_paper['1792'])
    # print(meta.paper_to_paper['2445'])
    # print(meta.paper_conf['1792'])
    meta.generate_metapath("data")
    # for p, lst in meta.paper_to_paper.items():
    #     print(p)
    #     print(lst)
    #     print('=====')
    #     # print(meta.paper_from_paper[])
    #
    # for id in meta.id_conf:
    #     print(type(id))

    # id = 77
    # global_id = meta.get_global_id(id, 3)
    # print(global_id)
    # print(meta.retrieve_type(global_id))
    # print(meta.retrieve_id(global_id))


if __name__ == "__main__":
    main()