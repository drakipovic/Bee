import re
import math


class CppLayoutFeatures(object):

    def __init__(self, train_source_code, test_source_code):
        self.train_source_code = train_source_code
        self.test_source_code = test_source_code

    def _ln(self, value, source_code_len):
        return math.log(float(value) / source_code_len) if value else 0

    def get_features(self):
        train_features, test_features = [], []

        for code in self.train_source_code:
            code_features = []
            code_features.append(self.tabs(code))
            code_features.append(self.spaces(code))
            code_features.append(self.whitespace_ratio(code))
            code_features.append(self.new_line_before_open_brace(code))
            code_features.append(self.tabs_lead_lines(code))

            train_features.append(code_features)
        
        for code in self.test_source_code:
            code_features = []
            code_features.append(self.tabs(code))
            code_features.append(self.spaces(code))
            code_features.append(self.whitespace_ratio(code))
            code_features.append(self.new_line_before_open_brace(code))
            code_features.append(self.tabs_lead_lines(code))

            test_features.append(code_features)

        return train_features, test_features
    
    def tabs(self, source_code):
        tabs_count = len(re.findall('\t', source_code))

        return self._ln(tabs_count, len(source_code))
    
    def spaces(self, source_code):
        spaces_count = len(re.findall(' ', source_code))

        return self._ln(spaces_count, len(source_code))
    
    def whitespace_ratio(self, source_code):
        whitespaces_count = len(re.findall('[\t| |\n]', source_code))

        return whitespaces_count / float(len(source_code) - whitespaces_count)
        
    #return wheter majority of { braces are written in new line or just after statement
    def new_line_before_open_brace(self, source_code):
        new_line_before_brace_count = len(re.findall('\n[\s]*\{', source_code))
        closed_paranthesis_before_brace_count = len(re.findall('\)[ |\t]*\{', source_code))

        return new_line_before_brace_count > closed_paranthesis_before_brace_count
    
    def tabs_lead_lines(self, source_code):
        lines = source_code.split('\n')

        tabs = 0
        spaces = 0
        for line in lines:
            if line.startswith('\t'):
                tabs += 1
            elif line.startswith(' '):
                spaces += 1
        
        return tabs > spaces
