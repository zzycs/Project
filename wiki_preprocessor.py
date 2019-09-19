#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a data pre-process class"""

import os


class WikiPreprocessor:

    def __init__(self, set_, overlap, seed=None):
        self.set_ = set_  # e.g. set1
        self.overlap = overlap
        # initialise node map and output files
        self.node_map = {}
        if seed is None:
            self.edge_list_output = open('wiki_output/' + overlap + '/' + set_ + '/' + set_ + '.edgelist', 'w')
            self.node_map_output = open('wiki_output/' + overlap + '/' + set_ + '/' + set_ + '.nodemap', 'w')
        else:
            self.seed = seed
            self.edge_list_output = open('wiki_output/' + overlap + '/' + set_ + '/' + seed.replace('.txt', '.edgelist'),
                                         'w')
            self.node_map_output = open('wiki_output/' + overlap + '/' + set_ + '/' + seed.replace('.txt', '.nodemap'), 'w')

    # convert single seed dataset to an edge list and a node map
    def convert_seed(self):
        with open('wiki_input/' + self.overlap + '/' + self.set_ + '/' + self.seed) as file:
            for line in file:
                self.output_edgelist(line)
        self.output_nodemap()

    # merge all seeds from a dataset into an edge list and a node map
    def merge_seeds(self):
        # open folder directory of set
        for seed in os.listdir('wiki_input/' + self.overlap + '/' + self.set_):
            if not seed.startswith('.'):
                # open seed file
                with open('wiki_input/' + self.overlap + '/' + self.set_ + '/' + seed) as file:
                    # create a new seed name in CopyB for same seed name of CopyA e.g. Danger UXB -> Danger UXB_Copy
                    if seed.endswith('_CopyB.txt'):
                        old_seed = '"' + seed.replace('_CopyB.txt', '') + '"'
                        new_seed = old_seed[:-1] + '_Copy"'
                        for line in file:
                            if old_seed in line:
                                line = line.replace(old_seed, new_seed)
                            self.output_edgelist(line)
                    else:
                        for line in file:
                            self.output_edgelist(line)
        self.output_nodemap()

    def output_edgelist(self, line):
        # remove quotation marks and line break, split by "," or ", "
        if '", "' in line:
            x, y = line[1:-2].strip().split('", "', 1)[:2]
        else:
            x, y = line[1:-2].strip().split('","', 1)[:2]
        self.mapping(x)
        self.mapping(y)
        # maybe remove duplicated edge before write
        self.edge_list_output.write(str(self.node_map[x]) + ' ' + str(self.node_map[y]) + '\n')

    def output_nodemap(self):
        for key in self.node_map:
            self.node_map_output.write(str(self.node_map[key]) + ': ' + key + '\n')

    def mapping(self, node):
        if self.node_map.get(node) is None:
            self.node_map[node] = self.node_map.__len__() + 9243


p1 = WikiPreprocessor('set2', '70', 'The 6th Day_CopyB.txt')
p1.convert_seed()
# p1.merge_seeds()
