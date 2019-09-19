#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a data pre-process class"""

import csv
import random
import networkx as nx
import numpy as np
from random import sample
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
import jellyfish


def read_venue_alignments(acm, dblp):
    venue_alignments = {}
    with open('paper_input/DBLP_ACM_matching_pair.csv') as matching_pairs:
        next(matching_pairs)
        for line in matching_pairs:
            dblp_id, acm_id = line.strip().split(',')[:2]
            acm_venue = acm.nodes(data=True)[acm_id]['venue']
            dblp_venue = dblp.nodes(data=True)[dblp_id]['venue']
            if venue_alignments.get(acm_venue) is None:
                venue_alignments[acm_venue] = dblp_venue
    print(venue_alignments)
    return venue_alignments


def build_graph(file='ACM', encoding='UTF-8'):
    print('Building graph for ' + file)
    G = nx.Graph()
    with open('paper_input/' + file + '.csv', encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # skip first line
        for row in csv_reader:
            G.add_node(row[0], title=row[1], authors=row[2].split(', '), venue=row[3], year=int(row[4]))
    add_edge(G)
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
    return G


def add_edge(G):
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x = nodes[i]
            y = nodes[j]
            if set(x[1]['authors']) & set(y[1]['authors']):
                G.add_edge(x[0], y[0])


acm = build_graph(file='ACM', encoding='UTF-8')
dblp = build_graph(file='DBLP2', encoding='ISO-8859-1')
print('Reading mapping...')
matching = {}
venue_alignments = {}
with open('paper_input/DBLP_ACM_matching_pair.csv') as matching_pairs:
    next(matching_pairs)
    for line in matching_pairs:
        dblp_id, acm_id = line.strip().split(',')[:2]
        if acm.has_node(acm_id) and dblp.has_node(dblp_id):
            matching[acm_id] = dblp_id
            acm_venue = acm.nodes(data=True)[acm_id]['venue']
            dblp_venue = dblp.nodes(data=True)[dblp_id]['venue']
            if venue_alignments.get(acm_venue) is None:
                venue_alignments[acm_venue] = dblp_venue
# print(venue_alignments)
positive_data_points = []
negative_data_points = []
true_labels = []
for x in list(acm.nodes(data=True)):
    for y in list(dblp.nodes(data=True)):
        # Attributes
        attrs = []
        # attrs.append(jellyfish.jaro_distance(x[1]['title'], y[1]['title']))
        attrs.append(jellyfish.jaro_winkler(x[1]['title'], y[1]['title']))
        # Author attribute
        if set(x[1]['authors']) & set(y[1]['authors']):
            attrs.append(1)
        else:
            attrs.append(0)
        # Year attribute
        if x[1]['year'] == y[1]['year']:
            attrs.append(1)
        else:
            attrs.append(0)
        # Venue attribute
        if venue_alignments[x[1]['venue']] == y[1]['venue']:
            attrs.append(1)
        else:
            attrs.append(0)
        if matching.get(x[0]) is not None and matching[x[0]] == y[0]:
            positive_data_points.append([attrs[0]])
        else:
            # if random.uniform(0, 1) > 0.9:
            if sum(attrs[1:]) == 3:
                negative_data_points.append([attrs[0]])
print('Total positive data points: ' + str(len(positive_data_points)))
print('Total negative data points: ' + str(len(negative_data_points)))
print('Evaluating...')
scores = []
for i in range(100):
    data_points = np.array(sample(positive_data_points, 500) + sample(negative_data_points, 500))
    true_labels = [1] * 500 + [0] * 500
    # data_points = np.array(positive_data_points + negative_data_points)
    # true_labels = [1] * len(positive_data_points) + [0] * len(negative_data_points)
    clf = XGBClassifier()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(clf, data_points, true_labels, scoring=scoring, cv=5)
    accuracy = results['test_accuracy'].mean()
    precision = results['test_precision'].mean()
    recall = results['test_recall'].mean()
    f1 = results['test_f1'].mean()
    scores.append((accuracy, precision, recall, f1))
print('Mean of accuracies: ' + str(sum([score[0] for score in scores]) / len(scores)))
print('Mean of precision: ' + str(sum([score[1] for score in scores]) / len(scores)))
print('Mean of recall: ' + str(sum([score[2] for score in scores]) / len(scores)))
print('Mean of f1: ' + str(sum([score[3] for score in scores]) / len(scores)))
