#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a data post-process class for paper"""

import pickle
import csv
import random
import jellyfish
import numpy as np
from random import sample
from ast import literal_eval
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class PaperPostprocessor:

    def __init__(self, emb_file, mapping_permutation, s, t, o):
        self.sep = s
        self.total = t
        self.option = o
        self.embeddings = np.load(emb_file)
        print(self.embeddings)
        self.emb_dict = {}
        self.attr_dict = {}
        for index, line in enumerate(self.embeddings):
            self.emb_dict[index] = line
            self.attr_dict[index] = self.extract_attrs(index)
        print('Total nodes: ' + str(len(self.emb_dict)))
        self.true_alignments = pickle.load(open(mapping_permutation, "rb"))
        print('Total true alignments: ' + str(len(self.true_alignments)))
        self.positive_data_points = []
        self.negative_data_points = []

    def write_data_points(self, path):
        print('Writing positive and negative data points...')
        positive_data_points_output = csv.writer(open(path + '_positive_data_points.csv', 'w'), delimiter=',')
        negative_data_points_output = csv.writer(open(path + '_negative_data_points.csv', 'w'), delimiter=',')
        for i in range(self.sep):
            for j in range(self.sep, self.total):
                if self.true_alignments.get(i) == j:
                    positive_data_points_output.writerow(self.embedding_combination(self.emb_dict[i], self.emb_dict[j]))
                else:
                    # if random.uniform(0, 1) > 0.99:
                    if self.attr_dict[i][1] == self.attr_dict[j][1] and self.attr_dict[i][2] == self.attr_dict[j][2] \
                            and jellyfish.jaro_winkler(self.attr_dict[i][0], self.attr_dict[j][0]) > 0.6:
                        negative_data_points_output.writerow(
                            self.embedding_combination(self.emb_dict[i], self.emb_dict[j]))

    def read_data_points(self, path):
        print('Reading positive and negative data points...')
        with open(path + '_positive_data_points.csv') as file:
            csv_reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for line in csv_reader:
                self.positive_data_points.append(line)
        with open(path + '_negative_data_points.csv') as file:
            csv_reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for line in csv_reader:
                self.negative_data_points.append(line)
        print('Total positive data points: ' + str(len(self.positive_data_points)))
        print('Total negative data points: ' + str(len(self.negative_data_points)))

    def evaluate(self, model='XGBoost', cv=True):
        data_points = np.array(sample(self.positive_data_points, 500) + sample(self.negative_data_points, 500))
        true_labels = [1] * 500 + [0] * 500
        if model == 'XGBoost':
            clf = XGBClassifier()
        else:
            clf = svm.SVC(gamma='auto')
        if cv:
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            results = cross_validate(clf, data_points, true_labels, scoring=scoring, cv=5)
            accuracy = results['test_accuracy'].mean()
            precision = results['test_precision'].mean()
            recall = results['test_recall'].mean()
            f1 = results['test_f1'].mean()
        else:
            X_train, X_test, y_train, y_test = train_test_split(data_points, true_labels, test_size=0.2, random_state=0)
            clf.fit(X_train, y_train)
            pred_labels = clf.predict(X_test)
            accuracy = accuracy_score(y_test, pred_labels)
            precision = precision_score(y_test, pred_labels)
            recall = recall_score(y_test, pred_labels)
            f1 = f1_score(y_test, pred_labels)
            # print(self.perf_measure(y_test, pred_labels))
        # print("%s, %s, %s, %s" % (accuracy, precision, recall, f1))
        return accuracy, precision, recall, f1

    @staticmethod
    def perf_measure(true_labels, pred_labels):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1 and true_labels[i] == 1:
                TP += 1
            elif pred_labels[i] == 0 and true_labels[i] == 0:
                TN += 1
            elif pred_labels[i] == 1 and true_labels[i] == 0:
                FP += 1
            elif pred_labels[i] == 0 and true_labels[i] == 1:
                FN += 1
        return TP, FN, FP, TN

    @staticmethod
    def write(data, file):
        with open(file, 'w') as output:
            for line in data:
                output.write(str(line) + '\n')

    @staticmethod
    def embedding_combination(emb1, emb2):
        return np.array(emb1) - np.array(emb2)

    def extract_attrs(self, id_):
        if id_ < self.sep:
            with open('paper_output/' + self.option + '_ACM.nodemap') as nodemap:
                for line in nodemap:
                    node_id, node_attr = line.strip().split(' ', 1)[:2]
                    if int(node_id) == id_:
                        return (literal_eval(node_attr)['title'], literal_eval(node_attr)['venue'],
                                literal_eval(node_attr)['year'])
        else:
            with open('paper_output/' + self.option + '_DBLP2.nodemap') as nodemap:
                for line in nodemap:
                    node_id, node_attr = line.strip().split(' ', 1)[:2]
                    if int(node_id) == id_:
                        return (literal_eval(node_attr)['title'], literal_eval(node_attr)['venue'],
                                literal_eval(node_attr)['year'])
        return None


option = 'new'
# p = PaperPostprocessor('paper_output/' + option + '_paper.emb',
#                        'paper_output/' + option + '_edges-mapping-permutation.txt',
#                        668, 1429, option)
p = PaperPostprocessor('paper_output/' + option + '_paper.emb',
                       'paper_output/' + option + '_edges-mapping-permutation.txt',
                       1629, 3738, option)
p.write_data_points('paper_output/' + option + '_paper')
p.read_data_points('paper_output/' + option + '_paper')
scores = []
for i in range(100):
    scores.append(p.evaluate(model='XGBoost', cv=True))
print('Mean of accuracies: ' + str(sum([score[0] for score in scores]) / len(scores)))
print('Mean of precision: ' + str(sum([score[1] for score in scores]) / len(scores)))
print('Mean of recall: ' + str(sum([score[2] for score in scores]) / len(scores)))
print('Mean of f1: ' + str(sum([score[3] for score in scores]) / len(scores)))
