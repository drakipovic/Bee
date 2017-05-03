import math

from feature_extraction import CppFeatureExtractor
from feature_extraction.lexical_features import CppLexicalFeatures


SOURCE_CODE_1 = """#include <cstdio>
                 #include <iostream>

                 #define MAX max(x, y)

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


def test_cpp_get_features_returns_correct_set_of_features():
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_4, SOURCE_CODE_5])

    features = cpp_lf.get_features()

    assert features == [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
                        [1, 1, 0, 1, 1, 1, 1, 0, math.log(1. / 44), 0, 0, 0, 0, 0, 0, 1]]

def test_cpp_lf_tokenize_returns_correct_freq_of_unigrams():
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_2])

    freq = cpp_lf.tokenize(SOURCE_CODE_2)

    assert freq == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]


def test_cpp_lf_keywords_freq_returns_correct_freq():
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_1])

    freq = cpp_lf.keyword_freq(SOURCE_CODE_1)

    assert freq == [math.log(1. / 869), math.log(1. / 869), math.log(1. / 869), math.log(1. / 869), 
                    math.log(1. / 869), math.log(1. / 869), math.log(1. / 869)]


def test_cpp_lf_keyword_returns_number_of_keyword_occured():
    cpp_lf = CppLexicalFeatures([SOURCE_CODE_3])

    keywords_count = cpp_lf.keywords(SOURCE_CODE_3)

    assert keywords_count == 8
