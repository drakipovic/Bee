import re
import math
from collections import Counter


class CppLexicalFeatures(object):

    def __init__(self, source_code, unigrams):
        self.source_code = source_code
        self.unigrams = unigrams
    
    def get_features(self):

        features = []
        for code in self.source_code:
            code_features = []
            code_features.extend(self.unigram_features(code))
            code_features.extend(self.keyword_freq(code))
            code_features.append(self.keywords(code))

            features.append(code_features)
        
        return features
    
    #returns frequency of word unigrams from source code
    def unigram_features(self, source_code):
        unigram_features = []
        tokens = re.split('\s+', source_code)
        frequencies = Counter(tokens)

        for unigram in self.unigrams:
            unigram_features.append(frequencies.get(unigram, 0))

        return unigram_features
    
    #counts number of occurences of keywords(if, while, else if, else, for, do, while)
    def keyword_freq(self, source_code):
        if_count = len(re.findall('[^(else)] if\s*\(.*?\)', source_code, re.DOTALL))
        for_count = len(re.findall('for\s*\(.*?\)', source_code, re.DOTALL))
        while_count = len(re.findall('while\s*\(.*?\)\s*\{.*?\}', source_code, re.DOTALL))
        do_count = len(re.findall('do\s*\{.*?\}', source_code, re.DOTALL))
        else_if_count = len(re.findall('else if\s*\(.*?\)', source_code, re.DOTALL))
        else_count = len(re.findall('else\s*\{.*?\}', source_code, re.DOTALL))
        switch_count = len(re.findall('switch\s*\(.*?\)', source_code, re.DOTALL))
        
        code_len = len(source_code)

        return [math.log(if_count / float(code_len)) if if_count else 0, math.log(for_count / float(code_len)) if for_count else 0, 
                math.log(while_count / float(code_len)) if while_count else 0, math.log(do_count / float(code_len)) if do_count else 0,
                math.log(else_count / float(code_len)) if else_count else 0, math.log(else_if_count / float(code_len)) if else_if_count else 0,
                math.log(switch_count / float(code_len)) if switch_count else 0]
    
    #returns number of unique keywords
    def keywords(self, source_code):
        cpp_keywords = ["alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
                        "case", "catch", "char", "char16_t", "char32_t", "class", "compl", "const",	"constexpr",
                        "const_cast", "continue", "decltype", "default", "delete", "do", "double", "dynamic_cast", 
                        "else", "enum", "explicit", "export", "extern",	"FALSE", "float", "for", "friend", "goto", 
                        "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
                        "nullptr", "operator", "or", "or_eq" ,"private" ,"protected" ,"public" ,"register", 
                        "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
                        "static_cast", "struct", "switch", "template", "this" ,"thread_local", "throw", "TRUE", "try",
                        "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", 
                        "volatile", "wchar_t", "while", "xor", "xor_eq", "override", "final"]

        tokens = re.split('\s+', source_code.replace('{', ' ').replace('(', ' '))

        keyword_counter = 0
        for token in tokens:
            if token in cpp_keywords:
                keyword_counter += 1
        
        return keyword_counter    