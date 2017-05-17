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


class CppSyntacticFeatures(object):

    def __init__(self, ast_train, ast_test):
        self.ast_train = ast_train
        self.ast_test = ast_test
    
    def get_features(self):
        features = []

        for ast_nodes in ast_train:
            #features.extend(self.average_node_depth(ast_nodes))
            pass
    
    def average_node_depth(self, ast_nodes):
        ast = ast_nodes.split('\n')
        depths = 
        for node in ast:
            data = node.split('\t')



    