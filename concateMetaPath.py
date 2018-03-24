import codecs
from utils.global_id import *

class concatMetaPath:

    def __init__(self):
        self.authorPair_pattern = dict() # K: author pair; V: list of pattern id
        self.pattern_id = dict()
        self.author_id = dict()
        self.author_group = dict()

    def build_dict(self, l1_file, group_file_dir, id_a_dir):
        p_id = 1
        with codecs.open(l1_file) as file:
            for line in file:
                toks = line.strip().split("\t")
                author_pair = (toks[0], toks[-1])
                pattern = tuple(toks[1:-1])

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

    def write_file_l1(self, file):
        for pair, pattern_dict in self.authorPair_pattern.items():
            a_pair = [retrieve_id(e) for e in pair]
            for pattern, cnt in pattern_dict.items():
                label = int(self.author_group[str(a_pair[0])] == self.author_group[str(a_pair[1])])
                lst = list(map(str, [a_pair[0], pattern, a_pair[1], cnt, label]))
                file.write('\t'.join(lst) + '\n')

    def write_pattern_id(self, file):
        for k, v in self.pattern_id.items():
            file.write(str(k) + '\t' + str(v) + '\n')

    def concatePath(self, prev_file, l1_file, out_file):
        '''
        :param prev_path_dir: the direction of path length - 1
        :param length: the length of path need to be generate
        '''
        for line_prev in prev_file:
            for line_l1 in l1_file:
                toks_prev = line_prev.strip().split("\t")
                toks_l1 = line_l1.strip().split("\t")
                if toks_prev[-3] == toks_l1[0]:
                    label = int(self.author_group[toks_prev[0]] == self.author_group[toks_l1[-3]])
                    new_path = toks_prev[:-2] + toks_l1[1:-2] + [int(toks_prev[-2]) * int(toks_l1[-2])] + [label]
                    new_path = list(map(str, new_path))
                    out_file.write('\t'.join(new_path) + '\n')




# def main():

meta = concatMetaPath()
# file = open("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt")
meta.build_dict("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt",
                "data/name-group.txt",
                "_reduced_dataset/output/id_author.txt")
# print(meta.authorPair_pattern)
# l1_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1", 'w')
# meta.write_pattern_id(file)
l1_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l1")
# l2_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2", 'w')
# print(meta.author_group)
# meta.write_file_l1(l1_file)
l2_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l2")
l3_file = open("_reduced_dataset/filter_venue_since_2005/pattern/meta-path_pattern_l3", 'w')
meta.concatePath(l2_file, l1_file, l3_file)
# meta.concatePath(l2_file, l1_file, l3_file)
#
# for pair in meta.author_pairs:
#     print( int(pair[0]) < int(pair[1]))


# if __name__ == "__main__":
#     main()







