# This is a processor file for graph only

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import centrality


class GraphProcessor:

    def __init__(self, set_):
        self.G = nx.Graph()
        if set_ is not None:
            self.set_ = set_

    def build_graph(self):
        print('Loading nodes')
        with open('output/' + self.set_ + '/' + self.set_ + '.nodemap') as nodemap:
            for line in nodemap:
                mapping, name = line.strip().split(': ', 1)[:2]
                self.G.add_node(int(mapping))
        print('Loading edges')
        with open('output/' + self.set_ + '/' + self.set_ + '.edgelist') as edgelist:
            for line in edgelist:
                a, b = line.strip().split(' ', 1)[:2]
                self.G.add_edge(int(a), int(b))

    def extract_betweenness_centrality(self):
        output = open('output/' + self.set_ + '/' + self.set_ + '_betweenness_centrality.csv', 'w')
        print('Calculating betweenness centrality')
        nodes = centrality.betweenness_centrality(self.G)
        for key in nodes:
            output.write(str(key) + ',' + str(nodes[key]) + '\n')

    def extract_closeness_centrality(self):
        output = open('output/' + self.set_ + '/' + self.set_ + '_closeness_centrality.csv', 'w')
        print('Calculating closeness centrality')
        nodes = centrality.closeness_centrality(self.G)
        for key in nodes:
            output.write(str(key) + ',' + str(nodes[key]) + '\n')

    def extract_degree_centrality(self):
        output = open('output/' + self.set_ + '/' + self.set_ + '_degree_centrality.csv', 'w')
        print('Calculating degree centrality')
        nodes = centrality.degree_centrality(self.G)
        for key in nodes:
            output.write(str(key) + ',' + str(nodes[key]) + '\n')

    @staticmethod
    def plot_betweenness_centrality(show=True, save=False):
        data = []
        for i in range(1, 5):
            with open('output/set' + str(i) + '/set' + str(i) + '_betweenness_centrality.csv') as csv:
                column = []
                for line in csv:
                    node, betweenness_centrality = line.strip().split(',', 1)[:2]
                    column.append(float(betweenness_centrality))
                data.append(column)
        fig, ax = plt.subplots()
        ax.set_title('Betweenness Centrality for all dataset')
        ax.boxplot(data, showfliers=False)
        if show:
            plt.show()
        if save:
            plt.savefig('image/betweenness_centrality.png', dpi=1200)
            plt.close()

    @staticmethod
    def plot_closeness_centrality(show=True, save=False):
        data = []
        for i in range(1, 5):
            with open('output/set' + str(i) + '/set' + str(i) + '_closeness_centrality.csv') as csv:
                column = []
                for line in csv:
                    node, closeness_centrality = line.strip().split(',', 1)[:2]
                    column.append(float(closeness_centrality))
                data.append(column)
        fig, ax = plt.subplots()
        ax.set_title('Closeness Centrality for all dataset')
        ax.boxplot(data, showfliers=False)
        if show:
            plt.show()
        if save:
            plt.savefig('image/closeness_centrality.png', dpi=1200)
            plt.close()

    @staticmethod
    def plot_degree_centrality(show=True, save=False):
        data = []
        for i in range(1, 14):
            with open('output/set' + str(i) + '/set' + str(i) + '_degree_centrality.csv') as csv:
                column = []
                for line in csv:
                    node, degree_centrality = line.strip().split(',', 1)[:2]
                    column.append(float(degree_centrality))
                data.append(column)
        fig, ax = plt.subplots()
        ax.set_title('Degree Centrality for all dataset')
        ax.boxplot(data, showfliers=False)
        if show:
            plt.show()
        if save:
            plt.savefig('image/degree_centrality.png', dpi=1200)
            plt.close()


g = GraphProcessor('set1')
g.build_graph()
g.extract_degree_centrality()
# g.extract_betweenness_centrality()
# g.extract_closeness_centrality()
# g.plot_betweenness_centrality()
# g.plot_closeness_centrality()
g.plot_degree_centrality()
