#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a data pre-process class"""

import csv
import pickle
import collections
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from ast import literal_eval
from collections import OrderedDict


class PaperPreprocessor:

    def __init__(self):
        self.true_alignment = {}
        self.venue_alignments = {}

    def read_venue_alignments(self):
        acm = nx.Graph()
        with open('paper_input/ACM.csv', encoding='UTF-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # skip first line
            for row in csv_reader:
                acm.add_node(row[0], title=row[1], venue=row[3], year=int(row[4]))
        dblp = nx.Graph()
        with open('paper_input/DBLP2.csv', encoding='ISO-8859-1') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # skip first line
            for row in csv_reader:
                dblp.add_node(row[0], title=row[1], venue=row[3], year=int(row[4]))
        with open('paper_input/DBLP_ACM_matching_pair.csv') as matching_pairs:
            next(matching_pairs)
            for line in matching_pairs:
                dblp_id, acm_id = line.strip().split(',')[:2]
                self.true_alignment[acm_id] = dblp_id
                acm_venue = acm.nodes(data=True)[acm_id]['venue']
                dblp_venue = dblp.nodes(data=True)[dblp_id]['venue']
                if self.venue_alignments.get(acm_venue) is None:
                    self.venue_alignments[acm_venue] = dblp_venue
        # print(self.venue_alignments)

    def filter_mapping(self, mapping_permutation):
        print('Writing mapping...')
        with open(mapping_permutation) as matching_pairs:
            matching = {}
            next(matching_pairs)
            for line in matching_pairs:
                dblp_id, acm_id = line.strip().split(',')[:2]
                dblp_mapping, dblp_title = self.extract_title(dblp_id)
                acm_mapping, acm_title = self.extract_title(acm_id)
                if dblp_title is not None and acm_title is not None:
                    matching[acm_mapping] = dblp_mapping
            pickle.dump(dict(OrderedDict(sorted(matching.items()))),
                        open('paper_output/new_edges-mapping-permutation.txt', 'wb'))

    def build_graph(self, file='ACM', first_label=0, encoding='UTF-8'):
        print('Building graph for ' + file)
        G = nx.Graph()
        with open('paper_input/' + file + '.csv', encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # skip first line
            for row in csv_reader:
                if self.venue_alignments.get(row[3]) is not None:
                    G.add_node(row[0], id=row[0], title=row[1], author=row[2], authors=row[2].split(', '),
                               venue=self.venue_alignments[row[3]], year=int(row[4]))
                else:
                    G.add_node(row[0], id=row[0], title=row[1], author=row[2], authors=row[2].split(', '),
                               venue=row[3], year=int(row[4]))
        self.add_edge(G, file)
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        # Remove unconnected components
        if not nx.is_connected(G):
            sub_graphs = list(nx.connected_component_subgraphs(G))
            main_graph = sub_graphs[0]
            # find the largest network in that list
            for sg in sub_graphs:
                if len(sg.nodes()) > len(main_graph.nodes()):
                    main_graph = sg
            G = main_graph
        # Convert node labels to continuous numbers
        G = nx.convert_node_labels_to_integers(G, first_label=first_label)
        return G
        # # Write edge list to file
        # nx.write_edgelist(G, 'paper_output/new_' + file + '.edgelist', data=False)
        # # Save node mapping and attributes
        # nodemap_output = open('paper_output/new_' + file + '.nodemap', 'w')
        # array = []
        # for node in sorted(list(G.nodes(data=True))):
        #     nodemap_output.write(str(node[0]) + ' ' + str(node[1]) + '\n')
        #     array.append([node[1]['title']])
        #     # array.append([node[1]['author']])
        #     # array.append([node[1]['title'], node[1]['venue'], node[1]['year']])
        # # Save in .npy file
        # np.save('paper_output/new_' + file, np.array(array))

    @staticmethod
    def merge():
        print('Merging...')
        f = open('paper_output/new_paper_combined_edges.txt', 'w')
        with open('paper_output/new_ACM.edgelist') as ACM:
            for line in ACM:
                f.write(line)
        with open('paper_output/new_DBLP2.edgelist') as DBLP2:
            for line in DBLP2:
                f.write(line)
        npy1 = np.load('paper_output/new_ACM.npy')
        npy2 = np.load('paper_output/new_DBLP2.npy')
        np.save('paper_output/new_paper_combined_attr', np.vstack((npy1, npy2)))

    def add_edge(self, G, file):
        nodes = list(G.nodes(data=True))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x = nodes[i]
                y = nodes[j]
                if set(x[1]['authors']) & set(y[1]['authors']):
                    # if file == 'ACM' and x[0] in self.true_alignment.keys() and y[0] in self.true_alignment.keys():
                    #     G.add_edge(x[0], y[0])
                    # if file == 'DBLP2' and \
                    #         x[0] in self.true_alignment.values() and y[0] in self.true_alignment.values():
                    G.add_edge(x[0], y[0])

    @staticmethod
    def extract_title(id_):
        if id_.isnumeric():
            with open('paper_output/new_ACM.nodemap') as nodemap:
                for l in nodemap:
                    mapping, a = l.strip().split(' ', 1)[:2]
                    attrs = literal_eval(a)
                    if attrs['id'] == id_:
                        return int(mapping), attrs['title']
        else:
            with open('paper_output/new_DBLP2.nodemap') as nodemap:
                for l in nodemap:
                    mapping, a = l.strip().split(' ', 1)[:2]
                    attrs = literal_eval(a)
                    if attrs['id'] == id_:
                        return int(mapping), attrs['title']
        return None, None


p1 = PaperPreprocessor()
p1.read_venue_alignments()
# p1.build_graph(file='DBLP2', first_label=668, encoding='ISO-8859-1')
# p1.merge()
# p1.filter_mapping('paper_input/DBLP_ACM_matching_pair.csv')
ACM = p1.build_graph(file='ACM', first_label=0, encoding='UTF-8')
acm_degreeCount = collections.Counter(sorted([d for n, d in ACM.degree()]))
acm_degree, acm_count = zip(*acm_degreeCount.items())
print(acm_degree)
print(acm_count)
print()
DBLP = p1.build_graph(file='DBLP2', first_label=1629, encoding='ISO-8859-1')
dblp_degreeCount = collections.Counter(sorted([d for n, d in DBLP.degree()]))
dblp_degree, dblp_count = zip(*dblp_degreeCount.items())
print(dblp_degree)
print(dblp_count)
colors = ['red', 'blue']
labels = ['ACM', 'DBLP2']
plt.plot(acm_degree, acm_count, label=labels[0], color=colors[0])
plt.plot(dblp_degree, dblp_count, label=labels[1], color=colors[1])
plt.xlim(0, 100)
plt.ylim(0, 100)
line_list = []
for i in range(2):
    line_list.append(
        lines.Line2D([], [], color=colors[i], label=labels[i]))
plt.legend(handles=line_list, loc='upper right')
plt.title("Degree Distribution")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()
