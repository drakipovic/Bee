import os


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

    def __init__(self, ast_train, ast_test):
        self.ast_train = ast_train
        self.ast_test = ast_test
    
    def get_features(self):
        train_features = []
        test_features = []

        for ast_nodes in self.ast_train:
            features = []
            features.extend(self.average_node_depth(ast_nodes))
            features.append(self.maximum_node_depth(ast_nodes))
            features.extend(self.keywords(ast_nodes))
            
            train_features.append(features)

        for ast_nodes in self.ast_test:
            features = []
            features.extend(self.average_node_depth(ast_nodes))
            features.append(self.maximum_node_depth(ast_nodes))
            features.extend(self.keywords(ast_nodes))

            test_features.append(features)
            
        return train_features, test_features

    def average_node_depth(self, ast_nodes):
        ast = ast_nodes.split('\n')
        depths = {node_type: [] for node_type in NODE_TYPES}

        for node in ast:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type in NODE_TYPES:
                depth = data[3]
                depths[node_type].append(int(depth))
            
        avg_depth = [sum(depths.get(node_type))/float(len(depths.get(node_type))) if depths.get(node_type) else 0 for node_type in sorted(depths)]
        
        return avg_depth
    
    def maximum_node_depth(self, ast_nodes):
        ast = ast_nodes.split('\n')

        max_depth = 0
        for node in ast:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type in NODE_TYPES:
                depth = int(data[3])

                if depth > max_depth:
                    max_depth = depth
        
        return max_depth

    #returns freq of cpp keywords
    def keywords(self, ast_nodes):
        ast = ast_nodes.split('\n')

        freq = {keyword: 0 for keyword in CPP_KEYWORDS}

        for node in ast:
            node = node.strip()

            data = node.split('\t')
            node_type = data[0]

            if node_type == 'KEYWORD':
                freq[data[4]] += 1
        
        return freq.values()




    