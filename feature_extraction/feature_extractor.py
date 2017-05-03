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
    def get_features(source_code, *args):
        return LanguageFeatureExtractor('cpp', source_code, *args).features


class LanguageFeatureExtractor(object):
    """Base class for extracting feature vector for different programming languages"""

    def __init__(self, language, source_code, *args):
        try:
            self.feature_extractor = LANGUAGE_EXTRACTORS[language](source_code, *args)
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

    def __init__(self, source_code, *args):
        self.source_code = source_code
        #we need this argument because we need to know all distinct unigrams before going into lexical extraction
        self.other_source_code = deepcopy(args[0])
        self.unigrams = self._get_word_unigrams()

    @property
    def lexical_features(self):
        return CppLexicalFeatures(self.source_code, self.unigrams).get_features()
    
    @property
    def layout_features(self):
        return CppLayoutFeatures(self.source_code).get_features()
    
    def _get_word_unigrams(self):
        self.other_source_code.extend(self.source_code)
    
        joined_sc = " ".join(self.other_source_code)

        tokens = re.split('\s+', joined_sc)
        frequencies = Counter(tokens)

        return frequencies.keys()


LANGUAGE_EXTRACTORS = {
    'cpp': CppFeatureExtractor
}