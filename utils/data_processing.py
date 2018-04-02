import codecs
import json
import os
from collections import Counter

paper_id = dict()
author_id = dict()
conf_id = dict()
paper_author = dict()
paper_conf = dict()
refernce_id = dict()
paper_paper = dict()
paper_conf = dict()
paper_cnt = 1
author_cnt = 1
conf_cnt = 1
added_paper = set()

target_author = set()
target_author_path = "../data/name-focus.txt"
with codecs.open(target_author_path, 'r', 'utf-8') as f:
    for line in f.readlines():
        line = line.strip().split("\t")
        target_author.add(line[0].replace('_', ' '))


path = "/Users/ruiyangwang/Desktop/ResearchProject/net_dblp"
# files = ["dblp-ref-0.json", "dblp-ref-3.json", "dblp-ref-2.json", "dblp-ref-1.json"]
# for file in files:
#     print("Delete others\n")
#     with codecs.open(path + "/" + file, 'r', 'utf-8') as j0:
#         with codecs.open(path + "/output.json", 'w', 'utf-8') as j1:
#             for line in j0.readlines():
#                 raw_data = json.loads(line)
#                 flag = 0
#                 if 'authors' not in raw_data:
#                     continue
#                 for author in raw_data['authors']:
#                     if author in target_author:
#                         flag = 1
#                         target_author[author] += 1
#                         break
#                 if flag == 0:
#                     continue
#                 j1.write(line)
#             j1.close()
#         j0.close()


# files = ["data_filter_venue_2005_focus.json"]
files = ["dblp-ref-0.json", "dblp-ref-3.json", "dblp-ref-2.json", "dblp-ref-1.json"]

for file in files:
    print("Loading {}".format(file))
    with codecs.open(path + "/" + file, 'r', 'utf-8') as j0:
        for line in j0.readlines():
            raw_data = json.loads(line)
            if 'authors' not in raw_data or 'id' not in raw_data or 'venue' not in raw_data or 'title' not in raw_data:
                continue
            if len(raw_data['venue']) == 0:
                continue
            for author in raw_data['authors']:
                if author not in target_author:
                    continue
                if author not in author_id:
                    author_id[author] = author_cnt
                    author_cnt += 1
            if raw_data['id'] not in refernce_id:
                refernce_id[raw_data['id']] = paper_cnt
                paper_cnt += 1
            if raw_data['venue'] not in conf_id:
                conf_id[raw_data['venue']] = conf_cnt
                conf_cnt += 1
            paper_author[refernce_id[raw_data['id']]] = []
            for author in raw_data['authors']:
                if author not in target_author:
                    continue
                paper_author[refernce_id[raw_data['id']]].append(author_id[author])
            paper_conf[refernce_id[raw_data['id']]] = conf_id[raw_data['venue']]
            paper_id[raw_data['title']] = refernce_id[raw_data['id']]
            added_paper.add(refernce_id[raw_data['id']])
            paper_conf[refernce_id[raw_data['id']]] = conf_id[raw_data['venue']]

    print("Complete loading {}".format(file))

for file in files:
    print("Reference\n")
    with codecs.open(path + "/" + file, 'r', 'utf-8') as j0:
        if 'authors' not in raw_data or 'id' not in raw_data or 'venue' not in raw_data or 'title' not in raw_data:
            continue
        if len(raw_data['venue']) == 0:
            continue
        for line in j0.readlines():
            raw_data = json.loads(line)
            if 'references' in raw_data:
                for refer in raw_data['references']:
                    if refer not in refernce_id or raw_data['id'] not in refernce_id:
                        continue
                    # if refer not in refernce_id:
                    #     refernce_id[refer] = paper_cnt
                    #     paper_cnt += 1
                    if refernce_id[raw_data['id']] not in paper_paper:
                        paper_paper[refernce_id[raw_data['id']]] = []
                    paper_paper[refernce_id[raw_data['id']]].append(refernce_id[refer])

print("Output {}".format("id_author.txt"))
# Output id_author
path += '/output/focus/full'
with codecs.open(path + "/id_author.txt", 'w', 'utf-8') as f:
    for author in author_id:
        f.write(str(author_id[author]) + "\t" + author + "\n")
    f.close()
print("Output {}".format("id_conf.txt"))
# Output id_conf
with codecs.open(path + "/id_conf.txt", 'w', 'utf-8') as f:
    for conf in conf_id:
        f.write(str(conf_id[conf]) + "\t" + str(conf) + "\n")
    f.close()
print("Output {}".format("paper_author.txt"))
# Output paper_author
with codecs.open(path + "/paper_author.txt", 'w', 'utf-8') as f:
    for pid in paper_author:
        for author in paper_author[pid]:
            f.write(str(pid) + "\t" + str(author) + "\n")
    f.close()
print("Output {}".format("paper_conf.txt"))
# Output paper_conf
with codecs.open(path + "/paper_conf.txt", 'w', 'utf-8') as f:
    for pid in paper_conf:
        f.write(str(pid) + "\t" + str(paper_conf[pid]) + "\n")
    f.close()
print("Output {}".format("paper.txt"))
# Output paper title
with codecs.open(path + "/paper.txt", 'w', 'utf-8') as f:
    for pid in paper_id:
        f.write(str(paper_id[pid]) + "\t" + str(pid) + "\n")
    f.close()
print("Output {}".format("paper_paper.txt"))
# Output paper reference
with codecs.open(path + "/paper_paper.txt", 'w', 'utf-8') as f:
    for pid in paper_paper:
        for pid2 in paper_paper[pid]:
            f.write(str(pid) + "\t" + str(pid2) + "\n")
    f.close()

