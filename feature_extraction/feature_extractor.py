import re
from collections import Counter, defaultdict, deque
from copy import deepcopy

from lexical_features import CppLexicalFeatures
from layout_features import CppLayoutFeatures
from syntactic_features import CppSyntacticFeatures
from keywords import NODE_TYPES


class LanguageNotSupportedException(Exception):
    pass


class FeatureExtractor(object):
    
    #returns lexical, layout and syntactic feature vector for a given language
    @staticmethod
    def get_features(train_source_code, test_source_code, ast_train=None, ast_test=None):
        return LanguageFeatureExtractor('cpp', train_source_code, test_source_code, ast_train, ast_test).features


class LanguageFeatureExtractor(object):
    """Base class for extracting feature vector for different programming languages"""

    def __init__(self, language, train_source_code, test_source_code, ast_train=None, ast_test=None):
        try:
            self.feature_extractor = LANGUAGE_EXTRACTORS[language](train_source_code, 
                                                                    test_source_code,
                                                                    ast_train,
                                                                    ast_test)
                                                
        except KeyError:
            raise LanguageNotSupportedException
        
    #returns feature vector for a language
    @property
    def features(self):
        train_lexical_features, test_lexical_features = self.feature_extractor.lexical_features
        train_layout_features, test_layout_features = self.feature_extractor.layout_features
        #train_syntactic_features, test_syntactic_features = self.feature_extractor.syntactic_features


        for idx, lf in enumerate(train_lexical_features):
            lf.extend(train_layout_features[idx])
            #lf.extend(train_syntactic_features[idx])
        
        for idx, lf in enumerate(test_lexical_features):
            lf.extend(test_layout_features[idx])
            #lf.extend(test_syntactic_features[idx])

        return train_lexical_features, test_lexical_features


class CppFeatureExtractor(object):

    def __init__(self, train_source_code, test_source_code, ast_train=None, ast_test=None):
        self.train_source_code = train_source_code
        self.test_source_code = test_source_code
        self.unigrams = self._get_word_unigrams()
        self.ast_train = ast_train
        self.ast_test = ast_test
        #self.node_bigrams, self.leaf_tf = self._create_tf()

    @property
    def lexical_features(self):
        return CppLexicalFeatures(self.train_source_code, self.test_source_code, self.unigrams,).get_features()
    
    @property
    def layout_features(self):
        return CppLayoutFeatures(self.train_source_code, self.test_source_code).get_features()
    
    @property
    def syntactic_features(self):
        return CppSyntacticFeatures(self.ast_train, self.ast_test, self.node_bigrams, self.leaf_tf).get_features()

    def _get_word_unigrams(self):
        joined_sc = " ".join(self.train_source_code)

        tokens = re.split('\s+', joined_sc)
        frequencies = Counter(tokens)

        return frequencies.keys()
    
    def _create_tf(self):
        bigrams = []
        leaf_tf = []
        
        for ast in self.ast_train:
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
                    leaf_tf.append(nodes[curr][1])
                
                for neighbor in edges.get(curr, []):
                    if nodes[neighbor][0] in NODE_TYPES and nodes[curr][0] in NODE_TYPES:
                        bigrams.append(nodes[curr][1] + '->' + nodes[neighbor][1])
                    
                    queue.append(neighbor)
        
        bigram_frequencies = Counter(bigrams)
        leaf_frequencies = Counter(leaf_tf)

        return bigram_frequencies.keys(), leaf_frequencies.keys()
        

LANGUAGE_EXTRACTORS = {
    'cpp': CppFeatureExtractor
}