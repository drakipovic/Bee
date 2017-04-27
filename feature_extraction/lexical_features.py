import re
import math
from collections import Counter


class CppLexicalFeatures(object):

    def __init__(self, source_code):
        self.source_code = source_code
    
    def get_features(self):
        features = []
        features.extend(self.keyword_freq())
        features.append(self.keywords())

        return features
    
    #returns frequency of word unigrams from source code
    def tokenize(self):
        tokens = re.split('\s+', self.source_code)
        frequencies = Counter(tokens)

        return frequencies.values()
    
    #counts number of occurences of keywords(if, while, else if, else, for, do, while)
    def keyword_freq(self):
        if_count = len(re.findall('[^(else)] if\s*\(.*?\)', self.source_code, re.DOTALL))
        for_count = len(re.findall('for\s*\(.*?\)', self.source_code, re.DOTALL))
        while_count = len(re.findall('while\s*\(.*?\)\s*\{.*?\}', self.source_code, re.DOTALL))
        do_count = len(re.findall('do\s*\{.*?\}', self.source_code, re.DOTALL))
        else_if_count = len(re.findall('else if\s*\(.*?\)', self.source_code, re.DOTALL))
        else_count = len(re.findall('else\s*\{.*?\}', self.source_code, re.DOTALL))
        switch_count = len(re.findall('switch\s*\(.*?\)', self.source_code, re.DOTALL))
        
        code_len = len(self.source_code)

        return [math.log(if_count / float(code_len)) if if_count else 0, math.log(for_count / float(code_len)) if for_count else 0, 
                math.log(while_count / float(code_len)) if while_count else 0, math.log(do_count / float(code_len)) if do_count else 0,
                math.log(else_count / float(code_len)) if else_count else 0, math.log(else_if_count / float(code_len)) if else_if_count else 0,
                math.log(switch_count / float(code_len)) if switch_count else 0]
    
    #returns number of unique keywords
    def keywords(self):
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

        tokens = re.split('\s+', self.source_code.replace('{', ' ').replace('(', ' '))

        keyword_counter = 0
        for token in tokens:
            if token in cpp_keywords:
                keyword_counter += 1
        
        return keyword_counter    