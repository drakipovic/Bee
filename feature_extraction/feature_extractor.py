import re
from collections import Counter
from copy import deepcopy

from lexical_features import CppLexicalFeatures
from layout_features import CppLayoutFeatures


class LanguageNotSupportedException(Exception):
    pass


class FeatureExtractor(object):
    
    #returns lexical, layout and syntactic feature vector for a given language
    @staticmethod
    def get_features(train_source_code, test_source_code):
        return LanguageFeatureExtractor('cpp', train_source_code, test_source_code).features


class LanguageFeatureExtractor(object):
    """Base class for extracting feature vector for different programming languages"""

    def __init__(self, language, train_source_code, test_source_code):
        try:
            self.feature_extractor = LANGUAGE_EXTRACTORS[language](train_source_code, test_source_code)
        except KeyError:
            raise LanguageNotSupportedException
        
    #returns feature vector for a language
    @property
    def features(self):
        lexical_features = self.feature_extractor.lexical_features
        layout_features = self.feature_extractor.layout_features
        
        features = lexical_features

        return features


class CppFeatureExtractor(object):

    def __init__(self, train_source_code, test_source_code):
        self.train_source_code = train_source_code
        self.test_source_code = test_source_code
        self.unigrams = self._get_word_unigrams()

    @property
    def lexical_features(self):
        return CppLexicalFeatures(self.train_source_code, self.test_source_code, self.unigrams).get_features()
    
    @property
    def layout_features(self):
        return CppLayoutFeatures(self.train_source_code).get_features()
    
    def _get_word_unigrams(self):
        joined_sc = " ".join(self.train_source_code)

        tokens = re.split('\s+', joined_sc)
        frequencies = Counter(tokens)

        return frequencies.keys()


LANGUAGE_EXTRACTORS = {
    'cpp': CppFeatureExtractor
}