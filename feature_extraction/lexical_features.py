import re
import math
import numpy as np
from collections import Counter


class CppLexicalFeatures(object):

    def __init__(self, train_source_code, test_source_code, unigrams=None, variable_names=None):
        self.train_source_code = train_source_code #list of source code for training
        self.test_source_code = test_source_code #list of source code for testing
        self.unigrams = unigrams
        self.variable_names = variable_names

    def _ln(self, value, source_code_len):
        return math.log(float(value) / source_code_len) if value else 0
    
    def get_features(self):

        train_features = []
        for code in self.train_source_code:
            code_features = []
            code_features.extend(self.unigram_features(code))
            code_features.extend(self.keyword_freq(code))
            code_features.append(self.keywords(code))
            code_features.append(self.ternary_operators(code))
            code_features.append(self.comments(code))
            code_features.append(self.literals(code))
            code_features.append(self.macros(code))
            code_features.append(self.functions(code))
            code_features.append(self.tokens(code))
            code_features.extend(self.line_length_measures(code))
            code_features.extend(self.function_parameters_measures(code))
            #code_features.extend(self.variable_freq(code))
            code_features.append(self.operators(code))

            train_features.append(code_features)
        
        test_features = []
        for code in self.test_source_code:
            code_features = []
            code_features.extend(self.unigram_features(code))
            code_features.extend(self.keyword_freq(code))
            code_features.append(self.keywords(code))
            code_features.append(self.ternary_operators(code))
            code_features.append(self.comments(code))
            code_features.append(self.literals(code))
            code_features.append(self.macros(code))
            code_features.append(self.functions(code))
            code_features.append(self.tokens(code))
            code_features.extend(self.line_length_measures(code))
            code_features.extend(self.function_parameters_measures(code))
            #code_features.extend(self.variable_freq(code))
            code_features.append(self.operators(code))

            test_features.append(code_features)
        
        return train_features, test_features
    
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

        return [self._ln(if_count, len(source_code)), self._ln(for_count, len(source_code)), 
                self._ln(while_count, len(source_code)), self._ln(do_count, len(source_code)),
                self._ln(else_count, len(source_code)), self._ln(else_if_count, len(source_code)), 
                self._ln(switch_count, len(source_code))]
    
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
        
        return self._ln(keyword_counter, len(source_code))
    
    #number of ternary operators
    def ternary_operators(self, source_code):

        ternary_count = len(re.findall(' \? ', source_code))

        return self._ln(ternary_count, len(source_code))
    
    #number of comments count
    def comments(self, source_code):

        comments_count = len(re.findall('/\*.*\*/', source_code, re.DOTALL))
        comments_count += len(re.findall('//', source_code, re.DOTALL))

        return self._ln(comments_count, len(source_code))

    #returns number of literals/constants
    def literals(self, source_code):

        literals_count = len(re.findall('#define [a-zA-Z][A-Za-z1-9|_]+ ', source_code))
        literals_count += len(re.findall('const (bool|int|float|long|long long|double|char|string) [a-zA-Z][A-Za-z0-9|_]+ =', source_code))

        return self._ln(literals_count, len(source_code))

    #returns number of word tokens
    def tokens(self, source_code):
        tokens = re.split('\s+', source_code)

        return self._ln(len(tokens), len(source_code))
    
    def functions(self, source_code):
        functions_count = len(re.findall('(std::)*(vector<|set<|list<|map<|unordered_map<|queue<|deque<|pair<|priority_queue<)*\s*(bool|int|float|long|long long|double|char|string)\s*[>]{0,1}\s*[a-zA-Z][A-Za-z0-9|_]*\(.*?\)', source_code, re.DOTALL))

        return self._ln(functions_count, len(source_code))

    def macros(self, source_code):
        macros = len(re.findall('#include|#define|#if|#ifndef|#ifdef|#else|#undef', source_code))

        return self._ln(macros, len(source_code))
    
    #returns avg and stddev of lines lengths
    def line_length_measures(self, source_code):
        lines_length = [len(line) for line in source_code.split('\n')]

        sum_lines_length = float(sum(lines_length))
        avg_line_length = sum_lines_length / len(lines_length)
        
        stddev_line_length = np.std(lines_length)

        return [avg_line_length, stddev_line_length]
        
    #returns avg and sttdev of function parameter number
    def function_parameters_measures(self, source_code):
        functions = re.findall('(?:const){0,1}(?:std::){0,1}(?:vector<|set<|list<|map<|unordered_map<|queue<|deque<|pair<|priority_queue<){0,1}[\s]*(?:bool|int|float|long|long long|double|char|string)[\s]*[>]{0,1}[\s]*[a-zA-Z][A-Za-z0-9|_]*\(.*?\)', source_code, re.DOTALL)

        parameters_length = []
        for function in functions:
            function_split = function.split('(')[1][:-1]
            parameters = function_split.split(',')
            if parameters[0] != '':
                parameters_length.append(len(parameters))
            else:
                parameters_length.append(0)
        
        avg_parameters = sum(parameters_length) / float(len(parameters_length)) if len(parameters_length) > 0 else 0
        stddev_parameters = np.std(parameters_length) if len(parameters_length) > 0 else 0

        return [avg_parameters, stddev_parameters]
    
    #returns freq of variable names
    def variable_freq(self, source_code):
        variables = re.findall('(?:const){0,1}(?:std::){0,1}(?:vector<|set<|list<|map<|unordered_map<|queue<|deque<|pair<|priority_queue<){0,1}[\s]*(?:int|float|long|long long|double|char|string)[\s]*[>]{0,1}[\s]*[a-zA-Z][A-Za-z0-9|_]*[\s]*[\(]{0,1}[=]*[\s]*[\"\'0-9a-zA-Z]*[\s]*[;|,|\)]', source_code, re.DOTALL)

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
        
        vn = {}
        for var_name in self.variable_names:
            vn[var_name] = 0
        
        for var_name in variable_names:
            if var_name in self.variable_names:
                vn[var_name] += 1

        return vn.values()

    def operators(self, source_code):
        addition = len(re.findall('[a-zA-Z0-9_]+[\s]*\+[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        substraction = len(re.findall('[a-zA-Z0-9_]+[\s]*\-[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        multiplication = len(re.findall('[a-zA-Z0-9_]+[\s]*\*[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        division = len(re.findall('[a-zA-Z0-9_]+[\s]*\/[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        modulo = len(re.findall('[a-zA-Z0-9_]+[\s]*%[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))

        increment = len(re.findall('(\+\+){1}[\s]*[a-zA-Z0-9_]+', source_code))
        increment += len(re.findall('[a-zA-Z0-9_]+[\s]*(\+\+){1}', source_code))
        decrement = len(re.findall('(\-\-){1}[\s]*[a-zA-Z0-9_]+', source_code))
        decrement += len(re.findall('[a-zA-Z0-9_]+[\s]*(\-\-){1}', source_code))

        equals = len(re.findall('[a-zA-Z0-9_]+[\s]*==[\s]*[a-zA-Z0-9_]+', source_code))
        not_equals = len(re.findall('[a-zA-Z0-9_]+[\s]*!=[\s]*[a-zA-Z0-9_]+', source_code))
        greater = len(re.findall('[a-zA-Z0-9_]+[\s]*[^<a-zA-Z\.]+>[\s]*[a-zA-Z0-9_]+', source_code))
        less = len(re.findall('[a-zA-Z0-9_]+[\s]*[^#include]<[\s]*[a-zA-Z0-9_]+', source_code))
        gte = len(re.findall('[a-zA-Z0-9_]+[\s]*>=[\s]*[a-zA-Z0-9_]+', source_code))
        lte = len(re.findall('[a-zA-Z0-9_]+[\s]*<=[\s]*[a-zA-Z0-9_]+', source_code))

        logical_and = len(re.findall('[a-zA-Z0-9_]+[\s]*&&[\s]*[a-zA-Z0-9_]+', source_code))
        logical_or = len(re.findall('[a-zA-Z0-9_]+[\s]*\|\|[\s]*[a-zA-Z0-9_]+', source_code))
        logical_not = len(re.findall('![\(a-zA-Z0-9_\)>!=\*+/%-&\| ]+', source_code))

        bitwise_and = len(re.findall('[a-zA-Z0-9_]+[\s]*\&[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        bitwise_or = len(re.findall('[a-zA-Z0-9_]+[\s]*\|[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        bitwise_xor = len(re.findall('[a-zA-Z0-9_]+[\s]*\^[\s]*[=]{0,1}[\s]*[a-zA-Z0-9_]', source_code))
        
        sc_len = len(source_code)

        operators = (addition + substraction + multiplication + division + modulo + increment +
                    decrement + equals + not_equals + greater + less + gte + lte + logical_and +
                    logical_or + logical_not + bitwise_and + bitwise_or + bitwise_xor)
        
        return self._ln(operators, sc_len) 