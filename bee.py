import os
import time
import argparse
import subprocess
from collections import defaultdict

import numpy as np

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest
from train.svm import SVM


basedir = os.path.abspath(os.path.dirname(__file__))
ALGORITHMS = {'random-forest': RandomForest, 'svm': SVM}


def extract_and_read_train_files(dataset_filename):
    print 'Extracting {}'.format(dataset_filename)

    os.system('mkdir dataset')
    os.system('unzip -q {} -d dataset'.format(dataset_filename))

    source_code = []
    labels = []
    ast_nodes = []

    for filename in sorted(os.listdir('dataset')):
        filename_split = filename.split('_')
        if 'cpp' not in filename_split[-1] and 'cxx' not in filename_split[-1]:
            continue
        
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        labels.append(author)
        with open(basedir + '/dataset/' + filename, 'r') as f:
            source_code.append(f.read())

        ast_nodes.append(subprocess.check_output(['java', '-jar', 'CodeSensor.jar', basedir + '/dataset/' + filename]))

    
    os.system('rm -rf dataset')

    return source_code, labels, ast_nodes


def create_accuracies_markdown_table(accuracies, n_trees):
    markdown = '| Variance Threshold '
    for nt in n_trees:
        markdown += '| {} trees '.format(nt)
    
    markdown += '\n'
    
    markdown += '| :---: '
    for nt in n_trees:
        markdown += '| --- '
    
    markdown += '\n'

    for vt in sorted(accuracies.keys()):
        markdown += '| **{}** '.format(vt)
        for acc, t in accuracies[vt]:
            markdown += '| {:.3f}% {:.1f}s '.format(float(acc), float(t))
        
        markdown += '\n'
    
    return markdown
    

def train(ml_algorithm, source_code, labels, ast_nodes):
    source_code = np.array(source_code)
    labels = np.array(labels)
    ast_nodes = np.array(ast_nodes)

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return

    start = time.time()

    mla = algorithm_type(variance_threshold=0.08)
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
                                                                        ast_nodes[train_indices],
                                                                        ast_nodes[test_indices])
        
        score = mla.train(train_features, test_features, labels[train_indices], labels[test_indices])
        print 'Score after {}th fold is {}'.format(i, score)
        accuracy += score
            
    print 'Final score: {}'.format(accuracy / float(k))
    end = time.time() - start
    print 'Execution time: {}'.format(end)
    print '-----------------------------------------------'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('--dataset', help="Dataset")

    args = parser.parse_args()

    dataset_filename = args.dataset

    if 'zip' in dataset_filename:
        source_code, labels, ast_nodes = extract_and_read_train_files(dataset_filename)
        train('random-forest', source_code, labels, ast_nodes)
    else:
        print 'File can only be zip!'