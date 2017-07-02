import os
import time
import argparse
from collections import defaultdict

import numpy as np

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest
from train.svm import SVM
from create_ast import create_ast_data


basedir = os.path.abspath(os.path.dirname(__file__))
ALGORITHMS = {'random-forest': RandomForest, 'svm': SVM}


def extract_and_read_source_files(train_filename=None, create_ast=False):
    train_filename = 'files/{}'.format(train_filename)

    os.system('mkdir train_files')
    os.system('unzip -q {} -d train_files'.format(train_filename))

    source_code = []
    labels = []
    ast_data = []

    for filename in sorted(os.listdir('train_files')):
        filename_split = filename.split('_')
        if 'cpp' not in filename_split[-1] and 'cxx' not in filename_split[-1]:
            continue
        
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        labels.append(author)
        with open(basedir + '/train_files/' + filename, 'r') as f:
            source_code.append(f.read())
    

    if create_ast:
        if not os.path.exists('parsed'):
            os.system('joern-parse train_files')
        ast_data = create_ast_data('train_files')     

    os.system('rm -rf train_files')

    return source_code, labels, ast_data



def train_and_predict(ml_algorithm, train_files, train_labels, test_files, test_labels):
    train_features, test_features = FeatureExtractor.get_features(train_files, test_files)

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return
    
    mla = algorithm_type()

    score = mla.fit_and_predict(train_features, test_features, train_labels, test_labels)

    print 'Score: {}'.format(score)


def train_kfold(ml_algorithm, source_code, labels, ast_data=None):
    source_code = np.array(source_code)
    labels = np.array(labels)
    ast_data = np.array(ast_data)

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return

    start = time.time()

    n_trees = [200, 250, 300, 350, 400, 500]
    vts = [0.03, 0.06, 0.08]

    for nt in n_trees:
        mla = algorithm_type(n_trees=nt)
        print 'nt: {}'.format(nt)
        k = 10
        code_per_author = 10
        accuracy = 0
        
        for i in range(k):
            test_indices = []
            it = code_per_author / k
            for j in range(it):
                test_indices.extend(np.array(range(i*it+j, len(source_code), code_per_author)))
                
            train_indices = []
            for sci in range(len(source_code)):
                if sci not in test_indices:
                    train_indices.append(sci)

            train_features, test_features = FeatureExtractor.get_features(source_code[train_indices],
                                                                            source_code[test_indices],
                                                                            ast_data[train_indices],
                                                                            ast_data[test_indices])
            

            score = mla.fit_and_predict(train_features, test_features, labels[train_indices], labels[test_indices])
            accuracy += score
                
        print 'Final score: {}'.format(accuracy / float(k))
        end = time.time() - start
        print 'Execution time: {}'.format(end)
        print '-----------------------------------------------'

    for nt in n_trees:
        for vt in vts:
            mla = algorithm_type(n_trees=nt, threshold=vt, svc=False)
            print 'nt: {} vt: {}'.format(nt, vt)
            k = 10
            code_per_author = 10
            accuracy = 0
            
            for i in range(k):
                test_indices = []
                it = code_per_author / k
                for j in range(it):
                    test_indices.extend(np.array(range(i*it+j, len(source_code), code_per_author)))
                    
                train_indices = []
                for sci in range(len(source_code)):
                    if sci not in test_indices:
                        train_indices.append(sci)

                train_features, test_features = FeatureExtractor.get_features(source_code[train_indices],
                                                                                source_code[test_indices],
                                                                                ast_data[train_indices],
                                                                                ast_data[test_indices])
                

                score = mla.fit_and_predict(train_features, test_features, labels[train_indices], labels[test_indices])
                print 'Score after {}th fold is {}'.format(i, score)
                accuracy += score
                
            print 'Final score: {}'.format(accuracy / float(k))
            end = time.time() - start
            
            print 'Execution time: {}'.format(end)
            print '-----------------------------------------------'



def fit(ml_algorithm, train_files, train_labels, name):
    train_features, _ = FeatureExtractor.get_features(train_files, [])

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return
    
    mla = algorithm_type()

    mla.fit(train_features, train_labels, name)


def get_probs(ml_algorithm, train_files, test_files, name):
    _, test_features = FeatureExtractor.get_features(train_files, test_files)

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return
    
    mla = algorithm_type()

    prob_ind, highest_score = mla.predict(test_features, name)
    return prob_ind, highest_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('--train_set', help="Train set")
    parser.add_argument('--test_set', help="Test set")

    args = parser.parse_args()

    source_code, labels, ast_data = extract_and_read_source_files(args.train_set, create_ast=True)
    train_kfold('random-forest', source_code, labels, ast_data)