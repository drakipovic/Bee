import math
import re
from collections import Counter

from feature_extraction import CppFeatureExtractor
from feature_extraction.lexical_features import CppLexicalFeatures


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


SOURCE_CODE_5 = """#include<iostream> if(bla){ cout << 'bla'; }"""


SOURCE_CODE_6 = """foo = (a > b) ? 1 : 0; bar = (b > a) ? 0 : 1"""


SOURCE_CODE_7 = """/* blablalba */
                   //jee"""


SOURCE_CODE_8 = """#define max(a, b) a > b ? a : b
                   #define MAX 1000000
                   
                   const int MAX_ITERATIONS = 100000
                   const int f()"""


def _create_unigrams(train, test):
    train.extend(test)

    joined_sc = " ".join(train)

    tokens = re.split('\s+', joined_sc)
    frequencies = Counter(tokens)

    return frequencies.keys()


def test_cpp_get_features_returns_correct_set_of_features():
    unigrams = _create_unigrams([SOURCE_CODE_4, SOURCE_CODE_5], [SOURCE_CODE_4])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_4, SOURCE_CODE_5], unigrams)

    features = cpp_lf.get_features()

    assert features == [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                        [1, 1, 0, 1, 1, 1, 1, 0, math.log(1. / len(SOURCE_CODE_5)), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

def test_cpp_lf_unigram_features_returns_correct_freq_of_unigrams():
    unigrams = _create_unigrams([SOURCE_CODE_2], [SOURCE_CODE_2])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_2], unigrams)

    freq = cpp_lf.unigram_features(SOURCE_CODE_2)

    assert freq == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]


def test_cpp_lf_keywords_freq_returns_correct_freq():
    unigrams = _create_unigrams([SOURCE_CODE_1], [SOURCE_CODE_1])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_1], unigrams)

    freq = cpp_lf.keyword_freq(SOURCE_CODE_1)

    assert freq == [math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1)), math.log(1. / len(SOURCE_CODE_1)), 
                    math.log(1. / len(SOURCE_CODE_1))]


def test_cpp_lf_keyword_returns_number_of_keyword_occured():
    unigrams = _create_unigrams([SOURCE_CODE_3], [SOURCE_CODE_3])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_3], unigrams)

    keywords_count = cpp_lf.keywords(SOURCE_CODE_3)

    assert keywords_count == 8


def test_cpp_lf_ternary_operator_number_returns_correct_number_of_ternary_op():
    unigrams = _create_unigrams([SOURCE_CODE_6], [SOURCE_CODE_6])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_6], unigrams)

    ternary_count = cpp_lf.ternary_operators(SOURCE_CODE_6)

    assert ternary_count == math.log(2. / len(SOURCE_CODE_6))


def test_cpp_lf_comments_returns_correct_number_of_comments():
    unigrams = _create_unigrams([SOURCE_CODE_7], [SOURCE_CODE_7])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_7], unigrams)

    comments_count = cpp_lf.comments(SOURCE_CODE_7)

    assert comments_count == math.log(2. / len(SOURCE_CODE_7))


def test_cpp_lf_literals_returns_correct_number_of_comments():
    unigrams = _create_unigrams([SOURCE_CODE_8], [SOURCE_CODE_8])
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_8], unigrams)

    literals_count = cpp_lf.literals(SOURCE_CODE_8)

    assert literals_count == math.log(2. / len(SOURCE_CODE_8))
