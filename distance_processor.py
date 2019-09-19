#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a data post-process class"""

import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from node import Node
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize


class DistanceProcessor:

    def __init__(self, set_, overlap):
        self.set_ = set_  # e.g. set1
        self.overlap = overlap
        self.seed_list = []  # list of seed object
        self.node_list = []  # list of node object
        self.seed = ''

        self.distance = ''  # name of distance algorithm, e.g. euclidean distance
        self.distance_matrix = None
        self.distance_list = []
        self.mds = None

        self.colors = ('blue', 'red', 'green')
        self.groups = ('p(Duplicated pair)', 'p(Copy A | Copy B, Others nodes)', 'p(Other seeds, Others nodes)')
        self.markers = ('*', '2', '3')

    # convert an embedding string to a list, split with space
    @staticmethod
    def convert_embedding(embedding_string):
        return [float(e) for e in embedding_string.split(' ')]

    def extract_deepwalk_seeds_only(self):
        for seed_name in os.listdir('wiki_input/' + self.overlap + '/' + self.set_):
            if not seed_name.startswith('.'):
                # extract seed name from file name
                if seed_name.endswith('_CopyA.txt'):
                    seed_name = seed_name.replace('_CopyA.txt', '')
                    self.seed = seed_name
                elif seed_name.endswith('_CopyB.txt'):
                    seed_name = seed_name.replace('B.txt', '')
                else:
                    seed_name = seed_name.replace('.txt', '')
                # extract seed mapping from node map
                seed_mapping = self.extract_node_mapping(seed_name)
                if seed_mapping is None:
                    print('Error: Seed (' + seed_name + ') not found.')
                else:
                    # extract embedding from file
                    with open(
                            'wiki_output/' + self.overlap + '/' + self.set_ + '/' + self.set_ + '_deepwalk.embeddings') as embeddings:
                        next(embeddings)  # skip first line
                        for embedding in embeddings:
                            if embedding.startswith(seed_mapping + ' '):  # e.g. 9 0.442661 -0.494164 0.00713700 ...
                                # truncate mapping
                                embedding_string = embedding.replace(seed_mapping + ' ', '', 1)
                                # create a node object and add to list
                                seed = Node(seed_name, seed_mapping, self.convert_embedding(embedding_string))
                                self.seed_list.append(seed)
                                break

    # add more nodes to embedding list and generate new distance matrix
    def extract_deepwalk_add_nodes(self, total_num=None):
        seeds = [s.mapping for s in self.seed_list]
        with open(
                'wiki_output/' + self.overlap + '/' + self.set_ + '/' + self.set_ + '_deepwalk.embeddings') as embeddings:
            next(embeddings)  # skip first line
            for embedding in embeddings:
                if total_num is not None and len(self.node_list) == total_num:
                    break
                else:
                    mapping, embedding = embedding.split(' ', 1)[:2]
                    if mapping in seeds:
                        continue
                    else:
                        node = Node(self.extract_node_name(mapping), mapping, self.convert_embedding(embedding))
                        self.node_list.append(node)
        # append seed list to node list
        self.node_list = self.seed_list + self.node_list

    # find node mapping from nodemap e.g. The Sweeney -> 2
    def extract_node_mapping(self, node_name):
        with open('wiki_output/' + self.overlap + '/' + self.set_ + '/' + self.set_ + '.nodemap') as nodemap:
            for line in nodemap:
                mapping, name = line.strip().split(': ', 1)[:2]
                if name == node_name:
                    return mapping
        return None

    # find node name from nodemap e.g. 2 -> The Sweeney
    def extract_node_name(self, node_mapping):
        with open('wiki_output/' + self.overlap + '/' + self.set_ + '/' + self.set_ + '.nodemap') as nodemap:
            for line in nodemap:
                mapping, name = line.strip().split(': ', 1)[:2]
                if mapping == node_mapping:
                    return name
        return None

    # calculate pairwise cosine similarity
    def extract_cosine_similarity(self):
        self.distance = 'cosine similarity'
        self.distance_matrix = cosine_similarity([node.embedding for node in self.node_list])

    def extract_euclidean_distances(self):
        self.distance = 'euclidean distances'
        self.distance_matrix = euclidean_distances([node.embedding for node in self.node_list])
        # normalization
        self.distance_matrix = normalize(self.distance_matrix)

    def extract_mds(self, n=2, s=4):
        embedding = MDS(n_components=n, random_state=s, dissimilarity='precomputed')
        self.mds = embedding.fit_transform(self.distance_matrix)

    def sort_distance(self):
        # only sort the first four lines, which refer to pairs that contain one or two seeds
        # change 4 to len(self.node_list), if you want sort all distances
        for i in range(0, 4):
            i_name = self.node_list[i].name
            for j in range(i + 1, len(self.distance_matrix[i])):
                j_name = self.node_list[j].name
                is_i_seed = i_name == self.seed or i_name == (self.seed + '_Copy')
                is_j_seed = j_name == self.seed or j_name == (self.seed + '_Copy')
                # classify pairs to groups
                if is_i_seed and is_j_seed:
                    group = 0
                elif is_i_seed or is_j_seed:
                    group = 1
                else:
                    group = 2
                pair_name = i_name + ',' + j_name
                self.distance_list.append((self.distance_matrix[i][j], group, pair_name))
        # csv_output = open('temp.csv', 'w')
        # for distance in sorted(self.distance_list):
        #     if distance[0] < 0.10:
        #         csv_output.write(str(distance[0]) + ',' + distance[2] + '\n')
        #         print(distance[0])
        #         print(distance[2])

    def plot_scatter(self, show=True, save=False):
        plt.scatter([m[0] for m in self.mds[4:]], [m[1] for m in self.mds[4:]], alpha=0.8, c='grey', marker='.',
                    label='Non-seeds')
        for i in range(0, 4):
            seed_name = self.node_list[i].name
            if self.seed == seed_name:
                color = self.colors[0]
                marker = self.markers[0]
            elif self.seed in seed_name:
                color = self.colors[1]
                marker = self.markers[0]
            else:
                color = self.colors[2]
                marker = self.markers[1]
            plt.scatter(self.mds[i][0], self.mds[i][1], alpha=0.8, c=color, marker=marker, label=seed_name)
        plt.title('Multidimensional scaling for ' + self.set_ + ' (' + self.seed + ')')
        plt.legend(loc=2, fontsize='small')
        if show:
            plt.show()
        if save:
            plt.savefig('image/MDS_' + self.set_ + '_' + str(len(self.node_list)) + 'nodes.png', dpi=1200)
            plt.close()

    def plot_distance_distribution(self, index):
        # plt.subplot(3, 5, index)
        count = 0
        for distance in sorted(self.distance_list, reverse=True):
            plt.plot(count, distance[0], marker=self.markers[distance[1]], markersize=5,
                     color=self.colors[distance[1]], label=self.groups[distance[1]])
            count += 1
        plt.ylim(0, 1)
        # plt.title(self.set_ + ': Copy pair = (' + self.seed + ' Copy A, ' + self.seed + ' Copy B)')
        line_list = []
        for i in range(len(self.groups)):
            line_list.append(
                lines.Line2D([], [], color=self.colors[i], label=self.groups[i], marker=self.markers[i], markersize=10))
        plt.legend(handles=line_list, loc='upper right')
        plt.savefig('test.png', dpi=250)
        plt.close()


# for index in range(1, 14):
p1 = DistanceProcessor('set' + str(1), '70')
p1.extract_deepwalk_seeds_only()
p1.extract_deepwalk_add_nodes(96)
p1.extract_cosine_similarity()
# p1.extract_euclidean_distances()
p1.sort_distance()
p1.plot_distance_distribution(index=1)
# p1.plot_scatter()
# colors = ('blue', 'red', 'green')
# groups = ('p(Copy pair)', 'p(Copy A | Copy B, Others nodes)', 'p(Other seeds, Others nodes)')
# markers = ('*', '2', '3')
# line_list = []
# for i in range(len(groups)):
#     line_list.append(
#         lines.Line2D([], [], color=colors[i], label=groups[i], marker=markers[i], markersize=10))
# plt.figlegend(handles=line_list, loc='lower center', ncol=5, labelspacing=0.)
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# fig.savefig('test.png', dpi=250)
