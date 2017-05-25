import math
import re
from collections import Counter, defaultdict

import numpy as np

from feature_extraction import CppFeatureExtractor
from feature_extraction.lexical_features import CppLexicalFeatures
from feature_extraction.layout_features import CppLayoutFeatures
from feature_extraction.syntactic_features import CppSyntacticFeatures


SOURCE_CODE_1 = """#include <cstdio>
                 #include <iostream>

                 #define MAX 10000

                 int main(){
                     if(1 == 1){
                         for(int i = 0; i < 10; ++i){
                             cout << i << endl;
                         }
                     }
                     else if(2 == 2){
                         while(i < 10){
                             cout << i << endl;
                             ++i;
                         }
                     }
                     else{
                         do{
                            ++i;
                            switch(i){
                                case 2: cout << i << endl;
                            };
                         } while(i < 10)
                     }
                 
                     return 0;
                 }"""


SOURCE_CODE_2 = """#include<iostream>
                   
                   int main(){

                       cout << 'Hello World' << endl;
                    
                       return 0;
                   }"""


SOURCE_CODE_3 = """if(i > 0) with() for()\{\} int if_bla long long bla short int a short short_bla"""


SOURCE_CODE_4 = """int main(){ cout << 'bla'; }"""


SOURCE_CODE_5 = """#include<iostream>
                   if(bla){ cout << 'bla'; }"""


SOURCE_CODE_6 = """foo = (a > b) ? 1 : 0; bar = (b > a) ? 0 : 1"""


SOURCE_CODE_7 = """/* blablalba */
                   //jee"""


SOURCE_CODE_8 = """#include<sctio>
                   #define max(a, b) a > b ? a : b
                   #define MAX 1000000
                   
                   const int MAX_ITERATIONS = 100000
                   const int f()"""


SOURCE_CODE_9 = """int f(float a, double b)
                   vector<int>bla(string bla, char bla)

                   std::vector<string> f(int iiiiaia, char ianiadnia)

                   queue bla(int, int)"""


SOURCE_CODE_10 = """\t\tif(a>0){}  """


SOURCE_CODE_11 = """if(a>0)
\t{
\tprint a;   
\tprint b;
}"""


SOURCE_CODE_12 = """if(a>0){
                        print a;
                    }"""


SOURCE_CODE_13 = """int bla = 0;
                    vector<int> bla;

                    int foo(int a=5, int b , int c)

                    string a = "dino";

                    vector<int>a(10);

                    std::vector<int>a(10);

                    for(int i = 0; i < 10; ++i){
                        int b = 0;
                    }"""


SOURCE_CODE_14 = """int a += 5; ++b; ++i; c -= 6;"""


AST_NODES_1 = """VAR_DECL	17:8	17:23	2	pair < int , int > pii
                 TYPE	17:8	17:8	3	pair < int , int >
                 TYPE_NAME	17:8	17:21	4	pair < int , int >
                 LEAF_NODE	17:8	17:8	5	pair
                 LEAF_NODE	17:13	17:13	5	<
                 LEAF_NODE	17:14	17:14	5	int
                 LEAF_NODE	17:17	17:17	5	,
                 LEAF_NODE	17:18	17:18	5	int
                 LEAF_NODE	17:21	17:21	5	>
                 NAME	17:23	17:23	3	pii
                 LEAF_NODE	17:23	17:23	4	pii
                 LEAF_NODE	17:26	17:26	2	;
                 SIMPLE_DECL	22:0	22:9	1	int n , int m , int l ;
                 INIT_DECL_LIST	22:0	22:0	2	int n , int m , int l
                 INIT_DECL_LIST	22:0	22:0	4	int n , int m , int l"""

AST_NODES_2 = """KEYWORD	39:1	39:1	4	if
                 LEAF_NODE	17:8	17:8	5	pair
                 LEAF_NODE	17:13	17:13	5	<
                 LEAF_NODE	17:14	17:14	5	int
                 LEAF_NODE	17:17	17:17	5	,
                 LEAF_NODE	17:18	17:18	5	int"""


def authors_per_leaf_node(ast_train):
    authors_freq_per_leaf_node = defaultdict(int)

    for ast in ast_train:
        ast_nodes = ast.split('\n')

        author_val = set()
        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')

            node_type = data[0]
            if node_type == 'LEAF_NODE':
                val = data[4]
                author_val.add(val)
        
        for v in author_val:
            authors_freq_per_leaf_node[v] += 1

    return authors_freq_per_leaf_node


def get_leaf_values(ast_train):
    leaf_values = {}

    for ast in ast_train:
        ast_nodes = ast.split('\n')

        for node in ast_nodes:
            node = node.strip()

            data = node.split('\t')

            node_type = data[0]
            if node_type == 'LEAF_NODE':
                leaf_values[data[4]] = 1

    return leaf_values


def _create_unigrams(train_code):
    joined_sc = " ".join(train_code)

    tokens = re.split('\s+', joined_sc)
    frequencies = Counter(tokens)

    return frequencies.keys()


def _get_variable_names(train_code):
        joined_sc = " ".join(train_code)

        variables = re.findall('(?:const){0,1}(?:std::){0,1}(?:vector<|set<|list<|map<|unordered_map<|queue<|deque<|pair<|priority_queue<){0,1}[\s]*(?:int|float|long|long long|double|char|string)[\s]*[>]{0,1}[\s]*[a-zA-Z][A-Za-z0-9|_]*[\s]*[\(]{0,1}[=]*[\s]*[\"\'0-9a-zA-Z]*[\s]*[;|,|\)]', joined_sc, re.DOTALL)

        variable_names = []
        for var in variables:
            if '=' in var:
                var_split = var.split('=')
                var_name = var_split[0].split()[1]
                var_name = var_name.strip()
                variable_names.append(var_name)

            elif '>' in var:
                var_split = var.split('>')
                if '(' in var_split[1]:
                    var_name = var_split[1].split('(')[0].strip()
                else:
                    var_name = var_split[1].split(';')[0]
                
                var_name = var_name.strip()
                variable_names.append(var_name)
            
            else:
                var_split = var.split()
                var_name = var_split[1].strip(',')
                var_name = var_name.strip(')')
                var_name = var_name.strip('(')
                variable_names.append(var_name)
        
        return variable_names


def test_cpp_layf_get_features_returns_correct_set_of_features():
     cpp_layf = CppLayoutFeatures([SOURCE_CODE_10], [])

     features, _ = cpp_layf.get_features()

     assert features == [[math.log(2. / len(SOURCE_CODE_10)), math.log(2. / len(SOURCE_CODE_10)),
                        4. / 9, False, True]]

def test_cpp_lf_unigram_features_returns_correct_freq_of_unigrams():
    unigrams = _create_unigrams([SOURCE_CODE_2])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_2], [], unigrams)

    freq = cpp_lf.unigram_features(SOURCE_CODE_2)

    assert freq == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]


def test_cpp_lf_keywords_freq_returns_correct_freq():
    unigrams = _create_unigrams([SOURCE_CODE_1])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_1], [], unigrams)

    freq = cpp_lf.keyword_freq(SOURCE_CODE_1)

    assert freq == [math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1))]


def test_cpp_lf_keyword_returns_number_of_keyword_occured():
    unigrams = _create_unigrams([SOURCE_CODE_3])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_3], [], unigrams)

    keywords_count = cpp_lf.keywords(SOURCE_CODE_3)

    assert keywords_count == math.log(8. / len(SOURCE_CODE_3))


def test_cpp_lf_ternary_operator_number_returns_correct_number_of_ternary_op():
    unigrams = _create_unigrams([SOURCE_CODE_6])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_6], [], unigrams)

    ternary_count = cpp_lf.ternary_operators(SOURCE_CODE_6)

    assert ternary_count == math.log(2. / len(SOURCE_CODE_6))


def test_cpp_lf_comments_returns_correct_number_of_comments():
    unigrams = _create_unigrams([SOURCE_CODE_7])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_7], [], unigrams)

    comments_count = cpp_lf.comments(SOURCE_CODE_7)

    assert comments_count == math.log(2. / len(SOURCE_CODE_7))


def test_cpp_lf_literals_returns_tabscorrect_number_of_comments():
    unigrams = _create_unigrams([SOURCE_CODE_8])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_8], [], unigrams)

    literals_count = cpp_lf.literals(SOURCE_CODE_8)

    assert literals_count == math.log(2. / len(SOURCE_CODE_8))


def test_cpp_lf_functions_returns_correct_number_of_functions():
    unigrams = _create_unigrams([SOURCE_CODE_9])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_9], [], unigrams)

    function_count = cpp_lf.functions(SOURCE_CODE_9)

    assert function_count == math.log(3. / len(SOURCE_CODE_9))


def test_cpp_lf_tokens_returns_correct_number_of_tokens():
    unigrams = _create_unigrams([SOURCE_CODE_5])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_5], [], unigrams)

    tokens_count = cpp_lf.tokens(SOURCE_CODE_5)

    assert tokens_count == math.log(6. / len(SOURCE_CODE_5))


def test_cpp_lf_macros_returns_correct_number_of_macros():
    unigrams = _create_unigrams([SOURCE_CODE_8])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_8], [], unigrams)

    macros_count = cpp_lf.macros(SOURCE_CODE_8)

    assert macros_count == math.log(3. / len(SOURCE_CODE_8))


def test_cpp_lf_line_length_measures_returns_correct_measures():
    unigrams = _create_unigrams([SOURCE_CODE_1])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_1], [], unigrams)

    line_length_measures = cpp_lf.line_length_measures(SOURCE_CODE_1)

    assert line_length_measures == [29.928571428571427, 13.071038245538359]


def test_cpp_lf_function_parameters_measures_returns_correct_measures():
    unigrams = _create_unigrams([SOURCE_CODE_9])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_9], [], unigrams)

    function_parameters_measures = cpp_lf.function_parameters_measures(SOURCE_CODE_9)

    assert function_parameters_measures == [2, 0]


def test_cpp_layf_tabs_count_return_correct_number_of_tabs():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_10], [])

    tabs_count = cpp_layf.tabs(SOURCE_CODE_10)

    assert tabs_count == math.log(2. / len(SOURCE_CODE_10))


def test_cpp_layf_spaces_count_return_correct_number_of_spaces():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_10], [])

    spaces_count = cpp_layf.spaces(SOURCE_CODE_10)

    assert spaces_count == math.log(2. / len(SOURCE_CODE_10))


def test_cpp_layf_whitespace_ratio_returns_correct_ratio():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_10], [])

    ratio = cpp_layf.whitespace_ratio(SOURCE_CODE_10)

    assert ratio == 4. / 9


def test_cpp_layf_new_line_before_braces_with_greater_number_of_new_lines_before_braces_returns_true():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_11], [])

    new_line_before_braces = cpp_layf.new_line_before_open_brace(SOURCE_CODE_11)

    assert new_line_before_braces


def test_cpp_layf_new_line_before_braces_with_greater_number_of_close_paranthesis_lines_before_braces_returns_false():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_12], [])

    new_line_before_braces = cpp_layf.new_line_before_open_brace(SOURCE_CODE_12)

    assert not new_line_before_braces


def test_cpp_layf_tabs_lead_line_with_greater_number_of_tabs_at_beginning_return_true():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_11], [])

    tabs_lead_lines = cpp_layf.tabs_lead_lines(SOURCE_CODE_11)

    assert tabs_lead_lines


def test_cpp_layf_tabs_lead_line_with_greater_number_of_spaces_at_beginning_return_false():
    cpp_layf = CppLayoutFeatures([SOURCE_CODE_12], [])

    tabs_lead_lines = cpp_layf.tabs_lead_lines(SOURCE_CODE_12)

    assert not tabs_lead_lines


def test_cpp_lf_variable_names_returns_correct_freq_of_variable_names():
    g_variable_names = _get_variable_names([SOURCE_CODE_13])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_13], [], variable_names=g_variable_names)
    

    variable_freq = cpp_lf.variable_freq(SOURCE_CODE_13)

    assert variable_freq == [4, 1, 1, 2, 2]


def test_cpp_lf_operators_returns_correct_number_of_operators():
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_14], [])

    operators = cpp_lf.operators(SOURCE_CODE_14)

    print operators

    assert operators == math.log(4. / len(SOURCE_CODE_14))



# def test_cpp_sf_average_node_depth_returns_correct_avg_depths():
#     cpp_sf = CppSyntacticFeatures([], [])

#     avg_depths = cpp_sf.average_node_depth(AST_NODES_1)

#     assert avg_depths == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 
#                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 2]


# def test_cpp_sf_max_node_depth_returns_correct_depth():
#     cpp_sf = CppSyntacticFeatures([], [])

#     max_depth = cpp_sf.maximum_node_depth(AST_NODES_1)

#     assert max_depth == 4


# def test_cpp_sf_keywords_returns_correct_keywords_freq():
#     cpp_sf = CppSyntacticFeatures([], [])

#     freq = cpp_sf.keywords(AST_NODES_2)

#     assert freq == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
#                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# def test_cpp_sf_leaf_values_freq_returns_correct_freq():
#     lv = get_leaf_values([AST_NODES_1])
#     cpp_sf = CppSyntacticFeatures([AST_NODES_1], [], lv)

#     freq = cpp_sf.leaf_values_freq(AST_NODES_1)

#     assert freq == [1, 2, 1, 1, 1, 1, 1]


# def test_cpp_sf_leaf_values_depths_returns_correct_avg_depth():
#     lv = get_leaf_values([AST_NODES_1])
#     cpp_sf = CppSyntacticFeatures([AST_NODES_1], [], lv)

#     avg_depths = cpp_sf.leaf_values_avg_depth(AST_NODES_1)

#     assert avg_depths == [4.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0]


# def test_cpp_sf_inverse_leaf_values_returns_correct_tfidf():
#     lv = get_leaf_values([AST_NODES_1, AST_NODES_2])
#     aplv = authors_per_leaf_node([AST_NODES_1, AST_NODES_2])
#     cpp_sf = CppSyntacticFeatures([AST_NODES_1, AST_NODES_2], [], lv, aplv)

#     inverse = cpp_sf.inverse_leaf_values_freq(AST_NODES_1)

#     assert inverse == [2, 2, 1, 1, 2, 1, 2]


# def test_cpp_sf_node_type_freq_returns_correct_tf():
#     cpp_sf = CppSyntacticFeatures([AST_NODES_1], [])

#     node_type_freq = cpp_sf.node_type_freq(AST_NODES_1)

#     assert node_type_freq == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
#                               0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]

# def test_cpp_sf_node_type_freq_inverse_returns_correct_tfidf():
#     cpp_sf = CppSyntacticFeatures([AST_NODES_1], [])

#     node_type_freq_inv = cpp_sf.node_type_freq_inverse(AST_NODES_1)

#     assert node_type_freq_inv == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
#                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]