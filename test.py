# This is a test file

import numpy as np
import networkx as nx
import csv
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import jellyfish
import nltk

def convert_embedding(embedding_string):
    return [float(e) for e in embedding_string.split(' ')]


def extract_node_mapping(set_, overlap, seed_name):
    with open('wiki_output/' + overlap + '/' + set_ + '/' + set_ + '.nodemap') as nodemap:
        for line in nodemap:
            mapping, name = line.strip().split(': ', 1)[:2]
            if name == seed_name:
                return mapping


def extract_embedding(set_, overlap, seed_mapping):
    # extract embedding from file
    with open('wiki_output/' + overlap + '/' + set_ + '/' + set_ + '_deepwalk.embeddings') as embeddings:
        next(embeddings)  # skip first line
        for embedding in embeddings:
            if embedding.startswith(seed_mapping + ' '):  # e.g. 9 0.442661 -0.494164 0.00713700 ...
                # truncate mapping
                return convert_embedding(embedding.replace(seed_mapping + ' ', '', 1))


overlaps = ['70', '50', '30']
seeds = ['Danger UXB', 'The 6th Day', 'Catherine Storr', 'John Monash', 'George Clooney', 'Kristen Wiig',
         'Maggie Smith', 'Emma Thompson', 'Val McDermid', 'Derek Jacobi', 'Simon Pegg', 'Peter Sellers',
         'Robertson Davies']
for i in range(len(seeds)):
    _set = 'set' + str(i + 1)
    seed_copy_a = seeds[i]
    seed_copy_b = seeds[i] + '_Copy'
    cosine_distance = []
    for _overlap in overlaps:
        embedding_1 = extract_embedding(_set, _overlap, extract_node_mapping(_set, _overlap, seed_copy_a))
        embedding_2 = extract_embedding(_set, _overlap, extract_node_mapping(_set, _overlap, seed_copy_b))
        cosine_distance.append(cosine_similarity([embedding_1], [embedding_2])[0][0])
    # plt.subplot(3, 5, i + 1)
    plt.plot(overlaps, cosine_distance, marker='o')
    plt.xlabel('Overlap (%)')
    # plt.ylabel('Cosine similarity')
    plt.ylim((0.5, 1.0))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test1.png', dpi=250)


# dictA = {}
# dictB = {}
# matching = {}
# with open('wiki_output/70/set2/The 6th Day_CopyA.nodemap') as copyA:
#     for line in copyA:
#         mapping, name = line.strip().split(': ', 1)[:2]
#         dictA[mapping] = name
# with open('wiki_output/70/set2/The 6th Day_CopyB.nodemap') as copyB:
#     for line in copyB:
#         mapping, name = line.strip().split(': ', 1)[:2]
#         dictB[mapping] = name
# for k1, a in dictA.items():
#     for k2, b in dictB.items():
#         if a == b:
#             matching[k1] = k2
# pickle.dump(matching, open('wiki_output/70/set2/The_6th_Day-mapping-permutation.txt', 'wb'))
# print(pickle.load(open('wiki_output/70/set2/The_6th_Day-mapping-permutation.txt', "rb"), encoding='latin1'))
#
# f = open('wiki_output/70/set2/The_6th_Day_combined_edges.txt', 'w')
# with open('wiki_output/70/set2/The 6th Day_CopyA.edgelist') as copyA:
#     for line in copyA:
#         f.write(line)
# with open('wiki_output/70/set2/The 6th Day_CopyB.edgelist') as copyB:
#     for line in copyB:
#         f.write(line)
