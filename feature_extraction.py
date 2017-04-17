from lexical_features import CppLexicalFeatures
from layout_features import CppLayoutFeatures


class LanguageNotSupportedException(Exception):
    pass


class FeatureExtractor(object):
    
    #returns lexical, layout and syntactic feature vector for a given language
    @staticmethod
    def get_features(language, source_code):
        return LanguageFeatureExtractor(language, source_code).features


class LanguageFeatureExtractor(object):
    """Base class for extracting feature vector for different programming languages"""

    def __init__(self, language, source_code):
        try:
            self.feature_extractor = LANGUAGE_EXTRACTORS[language](source_code)
        except KeyError:
            raise LanguageNotSupportedException
        
    #returns feature vector for a language
    @property
    def features(self):
        lexical_features = self.feature_extractor.lexical_features
        layout_features = self.feature_extractor.layout_features
        
        features = [lexical_features, layout_features]

        return features


class CppFeatureExtractor(object):

    def __init__(self, source_code):
        self.source_code = source_code

    @property
    def lexical_features(self):
        return CppLexicalFeatures().get_features(self.source_code)
    
    @property
    def layout_features(self):
        return CppLayoutFeatures().get_features(self.source_code)


LANGUAGE_EXTRACTORS = {
    'cpp': CppFeatureExtractor
}