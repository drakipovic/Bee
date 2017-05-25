import os
from collections import defaultdict


NODE_TYPES = ['SOURCE_FILE','FUNCTION_DEF','FUNCTION_NAME','RETURN_TYPE','CTOR_LIST','STATEMENTS',
              'INITIALIZER_ID', 'CTOR_INITIALIZER', 'NAME', 'PARAMETER_LIST', 'PARAMETER_DECL', 'ASSIGN',
              'CALLEE', 'ARGUMENT', 'CALL_TEMPLATE_LIST', 'INIT', 'VAR_DECL', 'POINTER', 'TYPE_SUFFIX',
              'SIMPLE_DECL','NAMESPACE_DEF','USING_DIRECTIVE','INCLUDE_DIRECTIVE','TEMPLATE_DECL_SPECIFIER',
              'SELECTION', 'ITERATION', 'KEYWORD', 'FOR_INIT', 'FOR_EXPR', 'JUMP_STATEMENT', 'DESTINATION',
              'CONDITION','LABEL','EXPR_STATEMENT','CTOR_EXPR','FUNCTION_CALL','CLASS_DEF', 'EQ_OPERATOR',
              'CLASS_NAME', 'TYPE_DEF','BASE_CLASSES', 'CLASS_CONTENT', 'TYPE_NAME', 'TYPE', 'BIT_OR_ELEM',
              'INIT_DECL_LIST', 'UNARY_EXPR', 'UNARY_OPERATOR', 'FIELD', 'EXPR', 'BIT_OR', 'BRACKETS',
              'CURLIES', 'SQUARES', 'AND', 'OR', 'COND_EXPR', 'ASSIGN_OP', 'LVAL', 'RVAL', 'REL_OPERATOR']

CPP_KEYWORDS = ["alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
                "case", "catch", "char", "char16_t", "char32_t", "class", "compl", "const",	"constexpr",
                "const_cast", "continue", "decltype", "default", "delete", "do", "double", "dynamic_cast", 
                "else", "enum", "explicit", "export", "extern",	"FALSE", "float", "for", "friend", "goto", 
                "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
                "nullptr", "operator", "or", "or_eq" ,"private" ,"protected" ,"public" ,"register", 
                "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
                "static_cast", "struct", "switch", "template", "this" ,"thread_local", "throw", "TRUE", 
                "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", 
                "volatile", "wchar_t", "while", "xor", "xor_eq", "override", "final", "try"]


class CppSyntacticFeatures(object):

    def __init__(self, ast_train, ast_test, ast_leaf_values=None, authors_per_leaf_val=None):
        self.ast_train = ast_train
        self.ast_test = ast_test
        self.ast_leaf_values = ast_leaf_values
        self.authors_per_leaf_val = authors_per_leaf_val
        self.authors_per_node_type = self._authors_per_node_type()

    def get_features(self):
        train_features = []
        test_features = []

        for ast in self.ast_train:
            features = []
            features.extend(self.average_node_depth(ast))
            features.append(self.maximum_node_depth(ast))
            features.extend(self.keywords(ast))
            #features.extend(self.node_type_freq(ast))
            #features.extend(self.node_type_freq_inverse(ast))
            #features.extend(self.leaf_values_freq(ast))
            #features.extend(self.leaf_values_avg_depth(ast))
            #features.extend(self.inverse_leaf_values_freq(ast))
            
            train_features.append(features)

        for ast in self.ast_test:
            features = []
            features.extend(self.average_node_depth(ast))
            features.append(self.maximum_node_depth(ast))
            features.extend(self.keywords(ast))
            #features.extend(self.node_type_freq(ast))
            #features.extend(self.node_type_freq_inverse(ast))
            #features.extend(self.leaf_values_freq(ast))
            #features.extend(self.leaf_values_avg_depth(ast))
            #features.extend(self.inverse_leaf_values_freq(ast))

            test_features.append(features)
            
        return train_features, test_features
    
    def _authors_per_node_type(self):
        authors_freq_per_node_type = {node_type: 0 for node_type in NODE_TYPES}

        for ast in self.ast_train:
            ast_nodes = ast.split('\n')

            author_val = set()
            for node in ast_nodes:
                node = node.strip()

                data = node.split('\t')

                node_type = data[0]
                if node_type in NODE_TYPES:
                    author_val.add(node_type)
            
            for v in author_val:
                authors_freq_per_node_type[v] += 1

        return authors_freq_per_node_type.values()

    def average_node_depth(self, ast):
        ast_nodes = ast.split('\n')
        depths = {node_type: [] for node_type in NODE_TYPES}

        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type in NODE_TYPES:
                depth = data[3]
                depths[node_type].append(int(depth))
            
        avg_depth = [sum(depths.get(node_type))/float(len(depths.get(node_type))) if depths.get(node_type) else 0 for node_type in sorted(depths)]
        
        return avg_depth
    
    def maximum_node_depth(self, ast):
        ast_nodes = ast.split('\n')

        max_depth = 0
        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type in NODE_TYPES:
                depth = int(data[3])

                if depth > max_depth:
                    max_depth = depth
        
        return max_depth

    #returns freq of cpp keywords
    def keywords(self, ast):
        ast_nodes = ast.split('\n')

        freq = {keyword: 0 for keyword in CPP_KEYWORDS}

        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type == 'KEYWORD':
                freq[data[4]] += 1
        
        return freq.values()

    def node_type_freq(self, ast):
        ast_nodes = ast.split('\n')
        node_tf = {node_type: 0 for node_type in NODE_TYPES}

        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type in NODE_TYPES:
                node_tf[node_type] += 1
        
        return node_tf.values()

    def node_type_freq_inverse(self, ast):
        node_type_tf = self.node_type_freq(ast)

        node_type_idftf = []
        for i, f in enumerate(self.authors_per_node_type):
            if f > 0:
                node_type_idftf.append((len(self.ast_train) / float(f)) * node_type_tf[i]) 
            else:
                node_type_idftf.append(0)

        return node_type_idftf

    def leaf_values_freq(self, ast):
        ast_nodes = ast.split('\n')
        leaf_values_freq = []

        leaf_values = defaultdict(int)
        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')
            
            node_type = data[0]
            if node_type == 'LEAF_NODE':
                leaf_values[data[4]] += 1
        
        for lv in self.ast_leaf_values:
            leaf_values_freq.append(leaf_values.get(lv, 0))
        
        return leaf_values_freq

    def inverse_leaf_values_freq(self, ast):
        leaf_values_freq = self.leaf_values_freq(ast)
        
        inverse_leaf_values_freq = []
        for i, aplf in enumerate(self.authors_per_leaf_val.values()):
            if aplf > 0:
                inverse_leaf_values_freq.append((len(self.ast_train) / float(aplf)) * leaf_values_freq[i])
            else:
                inverse_leaf_values_freq.append(0)

        return inverse_leaf_values_freq

    def leaf_values_avg_depth(self, ast):
        ast_nodes = ast.split('\n')
        leaf_value_depths = []

        depths = defaultdict(list)
        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')

            node_type = data[0]
            if node_type == 'LEAF_NODE':
                depths[data[4]].append(int(data[3]))
        
        for lv in self.ast_leaf_values:
            if depths.get(lv):
                leaf_value_depths.append(sum(depths.get(lv)) / float(len(depths.get(lv))))
            else:
                leaf_value_depths.append(0)
        
        return leaf_value_depths