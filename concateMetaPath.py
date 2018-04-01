import codecs
from utils.global_id import *
# from collections import OrderedSet

class concatMetaPath:

    def __init__(self):
        self.authorPair_pattern = dict() # K: author pair(not global id); V: list of pattern id
        self.pattern_id = dict()
        self.author_id = dict()
        self.author_group = dict()

    def build_dict(self, l1_full_path_dir, group_file_dir, id_a_dir):
        '''
        l1_full_path means the file that contains full meta-path information. e.g. A - P - A - P - A
        '''
        p_id = 1 # pattern id, increment it by 1 when find a new pattern
        with codecs.open(l1_full_path_dir) as file:
            for line in file:
                toks = line.strip().split("\t")
                author_pair = (retrieve_id(toks[0]), retrieve_id(toks[-1]))

                pattern = tuple([retrieve_type(e) for e in toks[1:-1]]) # e.g. P - A - P is a pattern
                # print(pattern)

                if not author_pair in self.authorPair_pattern:
                    self.authorPair_pattern[author_pair] = dict()

                if not pattern in self.pattern_id:
                    self.pattern_id[pattern] = p_id
                    p_id += 1

                curr_id = self.pattern_id[pattern]

                if curr_id in self.authorPair_pattern[author_pair]:
                    self.authorPair_pattern[author_pair][curr_id] += 1
                else:
                    self.authorPair_pattern[author_pair][curr_id] = 1

        with codecs.open(id_a_dir) as iafile:
            for line in iafile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.author_id[toks[1]] = toks[0]

        with codecs.open(group_file_dir) as agfile:
            for line in agfile:
                toks = line.strip().split("\t")
                toks[0] = toks[0].replace('_', ' ')
                if toks[0] in self.author_id:
                    self.author_group[self.author_id[toks[0]]] = toks[1]

    def write_file_l1(self, file_dir):
        file = open(file_dir, "w")
        # print(self.authorPair_pattern)
        for a_pair, pattern_dict in self.authorPair_pattern.items():
            # a_pair = [retrieve_id(e) for e in pair]
            for pattern, cnt in pattern_dict.items():
                label = int(self.author_group[str(a_pair[0])] == self.author_group[str(a_pair[1])])
                # print(pattern)
                lst = list(map(str, [a_pair[0], pattern, a_pair[1], cnt, label]))
                file.write('\t'.join(lst) + '\n')
        file.close()

    def write_pattern_id(self, out_dir):
        file = open(out_dir, "w")
        for k, v in self.pattern_id.items():
            file.write(str(k) + '\t' + str(v) + '\n')

    def concatePath(self, prev_file_dir, l1_file_dir, out_file_dir):
        '''
        :param prev_path_dir: the direction of path length - 1
        :param length: the length of path need to be generate
        '''

        # for line_prev in prev_file:
        #     print(line_prev)

        # load l1 file
        prev_file = open(prev_file_dir)
        l1_file = open(l1_file_dir)
        out_file = open(out_file_dir, 'w')

        l1_lines = []
        for line_l1 in l1_file:
            toks_l1 = line_l1.strip().split("\t")
            l1_lines.append(toks_l1)

        # prev_lines = []
        # for line_prev in prev_file:
        #     print("ininiini")
        #     toks_prev = line_prev.strip().split("\t")
        #     prev_lines.append(toks_prev)

        for line_prev in prev_file:
            # print(line_prev)
            toks_prev = line_prev.strip().split("\t")
            prev_a1, prev_path, prev_a2, prev_cnt = self.getToksInfo(toks_prev)
            # print('{0}--{1}--{2}--{3}'.format(prev_a1, prev_path, prev_a2, prev_cnt))
            for toks_l1 in l1_lines:
                # toks_l1 = list(toks_l1)
                l1_a1, l1_path, l1_a2, l1_cnt = self.getToksInfo(toks_l1)
                # print('{0}--{1}--{2}--{3}'.format(l1_a1, l1_path, l1_a2, l1_cnt))
                # if toks_prev[-3] == toks_l1[0]:
                #     label = int(self.author_group[toks_prev[0]] == self.author_group[toks_l1[-3]])
                #     new_path = toks_prev[:-2] + toks_l1[1:-2] + [int(toks_prev[-2]) * int(toks_l1[-2])] + [label]
                #     new_path = list(map(str, new_path))
                #     out_file.write('\t'.join(new_path) + '\n')

                prev_last_meta_path = self.get_last_meta_path(toks_prev)
                l1_meta_path = self.get_last_meta_path(toks_l1)
                sameAsPrev = self.same_meta_path(prev_last_meta_path, l1_meta_path)
                # check whether this l1 meta_path is as same as the last length 1 meta_path of the previous one

                if prev_a2 == l1_a1 and (not sameAsPrev):
                    label = int(self.author_group[prev_a1] == self.author_group[l1_a2])
                    new_path = [prev_a1] + prev_path + [prev_a2] + l1_path + [l1_a2] + [int(l1_cnt) * int(prev_cnt)] + [label]
                    new_path = list(map(str, new_path))
                    out_file.write('\t'.join(new_path) + '\n')


                # print(prev_last_meta_path)
                # print(l1_meta_path)
                # print('========')

                if prev_a2 == l1_a2 and int(prev_a1) < int(l1_a1) and (not sameAsPrev):
                    label = int(self.author_group[prev_a1] == self.author_group[l1_a1])
                    new_path = [prev_a1] + prev_path + [prev_a2] + l1_path + [l1_a1] + [int(l1_cnt) * int(prev_cnt)] + [label]
                    new_path = list(map(str, new_path))
                    out_file.write('\t'.join(new_path) + '\n')
        prev_file.close()
        l1_file.close()
        out_file.close()

    def getToksInfo(self, toks):
        # print(toks)
        # toks = list(toks)
        author1 = toks[0]
        inner_path = toks[1:-3]
        author2 = toks[-3]
        cnt = toks[-2]
        return author1, inner_path, author2, cnt

    def get_last_meta_path(self, toks):
        return toks[-5 : -2]

    def same_meta_path(self, m_p_1, m_p_2):
        m_p_1_reverse = list(reversed(m_p_1))
        return m_p_1 == m_p_2 or m_p_1_reverse == m_p_2

    def propagate_meta_path(self, l1_all_path, name_group_dir, id_author_dir, out_dir, length):
        # build dict
        self.build_dict(l1_all_path, name_group_dir, id_author_dir)
        # generate l1 file
        l1_file_dir = out_dir + "/meta-path_pattern_l1"
        print("generate meta-path with length {0}. File path: {1}".format(1, l1_file_dir))
        self.write_file_l1(l1_file_dir)
        # concate path
        for i in range(2, length + 1):
            write_file_dir = out_dir + "/meta-path_pattern_l" + str(i)
            prev_file_dir = out_dir + "/meta-path_pattern_l" + str(i - 1)
            print("generate meta-path with length {0}. File path: {1}".format(i, write_file_dir))
            self.concatePath(prev_file_dir, l1_file_dir, write_file_dir)
        print('Done')








l1_full_path_dir = "_reduced_dataset/pattern/meta_path_pattern_l1_all.txt"
name_group_dir = "data/name-group.txt"
id_author_dir = "_reduced_dataset/output/id_author.txt"
pattern_id_dir = "_reduced_dataset/filter_venue_since_2005/pattern_id.txt"
out_dir = "_reduced_dataset/pattern"



meta = concatMetaPath()
# meta.build_dict(l1_full_path_dir, name_group_dir, id_author_dir)
# meta.write_pattern_id(pattern_id_dir)
# print(meta.authorPair_pattern[(1, 4)])
meta.propagate_meta_path(l1_full_path_dir,name_group_dir,id_author_dir, out_dir, 5)
# file = open("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt")
# meta.build_dict("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt",
#                 "data/name-group.txt",
#                 "_reduced_dataset/output/id_author.txt")

# print(meta.author_group)
# print(meta.authorPair_pattern)
# l1_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1", 'w')
# meta.write_pattern_id(file)
# l1_file_dir = "_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1_test"
# l2_file_dir = "_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2_test"
# l3_file_dir = "_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l3_test"
# print(meta.author_group)
# meta.write_file_l1(l1_file_dir)
# l2_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2")
# l3_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l3", 'w')
# meta.concatePath(l1_file_dir, l1_file_dir, l2_file_dir)
# meta.concatePath(l2_file_dir, l1_file_dir, l3_file_dir)
#
# for pair in meta.author_pairs:
#     print( int(pair[0]) < int(pair[1]))


# if __name__ == "__main__":
#     main()







