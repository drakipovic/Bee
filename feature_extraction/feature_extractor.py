import re
from collections import Counter, defaultdict
from copy import deepcopy

from lexical_features import CppLexicalFeatures
from layout_features import CppLayoutFeatures
from syntactic_features import CppSyntacticFeatures


class LanguageNotSupportedException(Exception):
    pass


class FeatureExtractor(object):
    
    #returns lexical, layout and syntactic feature vector for a given language
    @staticmethod
    def get_features(train_source_code, test_source_code, ast_train=None, ast_test=None):
        return LanguageFeatureExtractor('cpp', train_source_code, test_source_code).features


class LanguageFeatureExtractor(object):
    """Base class for extracting feature vector for different programming languages"""

    def __init__(self, language, train_source_code, test_source_code, ast_train=None, ast_test=None):
        try:
            self.feature_extractor = LANGUAGE_EXTRACTORS[language](train_source_code, test_source_code)
                                                
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
        self.variable_names = self._get_variable_names()
        # self.ast_train = ast_train
        # self.ast_test = ast_test
        # self.ast_leaf_values = self._get_leaf_values(ast_train)
        # self.authors_freq_per_leaf_node = self._authors_per_leaf_node(ast_train)

    @property
    def lexical_features(self):
        return CppLexicalFeatures(self.train_source_code, self.test_source_code, 
                                    self.unigrams, self.variable_names).get_features()
    
    @property
    def layout_features(self):
        return CppLayoutFeatures(self.train_source_code, self.test_source_code).get_features()
    
    @property
    def syntactic_features(self):
        return CppSyntacticFeatures(self.ast_train, self.ast_test, 
                                    self.ast_leaf_values, self.authors_freq_per_leaf_node).get_features()

    def _get_leaf_values(self, ast_train):
        leaf_values = {}
        
        for ast in ast_train:
            ast_nodes = ast.split('\n')

            for node in ast_nodes:
                node = node.strip()
                
                data = node.split('\t')
                
                node_type = data[0]
                if node_type == 'LEAF_NODE':
                    leaf_values[data[4]] = 1
        
        return leaf_values
    
    def _authors_per_leaf_node(self, ast_train):
        authors_freq_per_leaf_node = defaultdict(int)

        for ast in ast_train:
            ast_nodes = ast.split('\n')

            author_val = set()
            for node in ast_nodes:
                node = node.strip()

                data = node.split('\t')

                node_type = data[0]
                if node_type == 'LEAF_NODE':
                    val = data[4]
                    author_val.add(val)
            
            for v in author_val:
                authors_freq_per_leaf_node[v] += 1

        return authors_freq_per_leaf_node

    def _get_word_unigrams(self):
        joined_sc = " ".join(self.train_source_code)

        tokens = re.split('\s+', joined_sc)
        frequencies = Counter(tokens)

        return frequencies.keys()
    
    def _get_variable_names(self):
        joined_sc = " ".join(self.train_source_code)

        variables = re.findall('(?:const){0,1}(?:std::){0,1}(?:vector<|set<|list<|map<|unordered_map<|queue<|deque<|pair<|priority_queue<){0,1}[\s]*(?:int|float|long|long long|double|char|string)[\s]*[>]{0,1}[\s]*[\*]{0,2}[a-zA-Z][A-Za-z0-9|_]*[\s]*[\(]{0,1}[=]{0,1}[\s]*[\"\'0-9a-zA-Z]*[\s]*[;|,|\)]', joined_sc, re.DOTALL)

        variable_names = []
        for var in variables:
            if '=' in var:
                var_split = var.split('=')
                try:
                    var_name = var_split[0]
                except IndexError:
                    continue
                
                try:
                    var_name = var_split[0].split()[1]
                except IndexError:
                    continue
                
                var_name = var_name.strip()
                variable_names.append(var_name)

            elif '>' in var:
                var_split = var.split('>')
                if '(' in var_split[1]:
                    try:
                        var_name = var_split[1].split('(')[0].strip()
                    except IndexError:
                        continue

                else:
                    try:
                        var_name = var_split[1].split(';')[0]
                    except IndexError:
                        continue

                var_name = var_name.strip()
                variable_names.append(var_name)
            
            else:
                var_split = var.split()
                try:
                    var_name = var_split[1]
                except IndexError:
                    continue
                
                try:
                    var_name = var_split[1].strip(',')
                except IndexError:
                    continue
                    
                var_name = var_name.strip(')')
                var_name = var_name.strip('(')
                variable_names.append(var_name)
        
        return variable_names


LANGUAGE_EXTRACTORS = {
    'cpp': CppFeatureExtractor
}