import os
from collections import defaultdict, deque, Counter

from keywords import NODE_TYPES, CPP_KEYWORDS


class CppSyntacticFeatures(object):

    def __init__(self, ast_train, ast_test, node_bigrams, leaf_tf):
        self.ast_train = ast_train
        self.ast_test = ast_test
        self.node_bigrams = node_bigrams
        self.leaf_tf = leaf_tf

    def get_features(self):
        train_features = []
        test_features = []

        for ast in self.ast_train:
            features = self.node_bigrams_tf(ast)
            features.extend(self.node_type_tf(ast))
            features.extend(self.leaf_values_tf(ast))

            train_features.append(features)
        
        for ast in self.ast_test:
            features = self.node_bigrams_tf(ast)
            features.extend(self.node_type_tf(ast))
            features.extend(self.leaf_values_tf(ast))

            test_features.append(features)
            
        return train_features, test_features

    
    def node_bigrams_tf(self, ast):
        bigrams = []
        edges = ast[0]
        nodes = ast[1]

        root = min(edges.keys())
        
        queue = deque([root])
        visited = []

        while len(queue) > 0:
            curr = queue.popleft()

            if curr in visited:
                continue
            
            visited.append(curr)
            
            for neighbor in edges.get(curr, []):
                if nodes[neighbor][0] in NODE_TYPES and nodes[curr][0] in NODE_TYPES:
                    bigrams.append(nodes[curr][1] + '->' + nodes[neighbor][1])

                queue.append(neighbor)
        
        frequencies = Counter(bigrams)
        
        nb_tf = []
        for node_bigram in self.node_bigrams:
            nb_tf.append(frequencies.get(node_bigram, 0))
        
        return nb_tf

    def node_type_tf(self, ast):
        node_type_tf = {nt: 0 for nt in NODE_TYPES}

        edges = ast[0]
        nodes = ast[1]

        root = min(edges.keys())
        
        queue = deque([root])
        visited = []

        while len(queue) > 0:
            curr = queue.popleft()

            if curr in visited:
                continue
            visited.append(curr)
            if edges.get(curr, None) and nodes[curr][0] in NODE_TYPES:
                node_type_tf[nodes[curr][0]] += 1
            
            for neighbor in edges[curr]:
                queue.append(neighbor)
        
        return node_type_tf.values()

    def leaf_values_tf(self, ast):
        leaf_values = []
        edges = ast[0]
        nodes = ast[1]

        root = min(edges.keys())
        
        queue = deque([root])
        visited = []

        while len(queue) > 0:
            curr = queue.popleft()

            if curr in visited:
                continue
            
            visited.append(curr)
            
            if edges.get(curr, []) == []:
                leaf_values.append(nodes[curr][1])
                continue
            
            for neighbor in edges[curr]:
                queue.append(neighbor)
        
        frequencies = Counter(leaf_values)
        leaf_value_tf = []
        
        for l in self.leaf_tf:
            leaf_value_tf.append(frequencies.get(l, 0))
        
        return leaf_value_tf