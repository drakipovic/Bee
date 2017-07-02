import os
import csv
from collections import defaultdict



def create_ast_data(dirname):
    edges_list = []
    nodes_list = []

    dirname = 'parsed/{}'.format(dirname)

    for filename in os.listdir(dirname):
        if filename == 'edges.csv' or filename == 'nodes.csv':
            continue
        
        edges = defaultdict(list)
        with open('{}/{}/edges.csv'.format(dirname, filename), 'r') as edges_csv:
            edges_reader = csv.reader(edges_csv, delimiter='\t')

            f = 0
            for edge in edges_reader:
                if f == 0:
                    f += 1
                    continue
                
                edges[int(edge[0])].append(int(edge[1]))

        nodes = defaultdict(tuple)
        with open('{}/{}/nodes.csv'.format(dirname, filename), 'r') as nodes_csv:
            nodes_reader = csv.reader(nodes_csv, delimiter='\t')

            f = 0
            for node in nodes_reader:
                if f == 0:
                    f += 1
                    continue
            
                nodes[int(node[1])] = (node[2], node[3].replace(' ', ''))
        
        edges_list.append(edges)
        nodes_list.append(nodes)

    data = []
    for e, n in zip(edges_list, nodes_list):
        data.append((e, n))

    print 'ast data created'
    return data