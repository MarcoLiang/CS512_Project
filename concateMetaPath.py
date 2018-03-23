import codecs
from utils.global_id import *

class concatMetaPath:

    def __init__(self):
        self.authorPair_pattern = dict() # K: author pair; V: list of pattern id
        self.pattern_id = dict()

    def build_dict(self, l1_file):
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
                    self.authorPair_pattern[author_pair][curr_id] = 0

                #
                #
                # self.author_pairs.add(author_pair)
                #
                # if not pattern in self.pattern_id:
                #     self.pattern_id[pattern] = p_id
                #     p_id += 1

    def concatePath(self, prev_path_dir, length):
        '''
        :param prev_path_dir: the direction of path length - 1
        :param length: the length of path need to be generate
        '''
        assert length > 1



# def main():

meta = concatMetaPath()
file = open("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt")
meta.build_dict("_reduced_dataset/pattern/meta_path_pattern_l1_all.txt")
print(meta.authorPair_pattern)


#
# for pair in meta.author_pairs:
#     print( int(pair[0]) < int(pair[1]))


# if __name__ == "__main__":
#     main()







