import time
import argparse
from zipfile import ZipFile
from collections import defaultdict

import numpy as np

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest

ALGORITHMS = {'random-forest': RandomForest}


def extract_and_read_train_files(dataset_filename):
    print 'Extracting {}'.format(dataset_filename)

    source_code = []
    labels = []

    zip_file = ZipFile(dataset_filename, "r")
    for filename in sorted(zip_file.namelist()):
        filename_split = filename.split('_')
        if 'cpp' not in filename_split[-1] and 'cxx' not in filename_split[-1]:
            continue
        
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        labels.append(author)

        source_code.append(zip_file.read(filename))
    
    return source_code, labels


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
    

def train(ml_algorithm, source_code, labels):
    source_code = np.array(source_code)
    labels = np.array(labels)

    try:
        algorithm_type = ALGORITHMS[ml_algorithm]
    except KeyError:
        print 'Algorithm type not valid!'
        return

    n_trees = [50, 100, 150, 200, 250, 300, 350, 400, 500, 800, 1000, 2000]
    variance_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    data = defaultdict(list)
    
    for nt in n_trees:
        for vt in variance_threshold:
            start = time.time()
            print 'N of Trees: {} | Variance Threshold: {}'.format(nt, vt)

            mla = algorithm_type(n_trees=nt, variance_threshold=vt)
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
                
                train_features, test_features = FeatureExtractor.get_features(source_code[train_indices], source_code[test_indices])
                
                score = mla.train(train_features, test_features, labels[train_indices], labels[test_indices])
                accuracy += score
            
            print 'Final score: {}'.format(accuracy / float(k))
            end = time.time() - start
            print 'Execution time: {}'.format(end)
            print '-----------------------------------------------'
            data[vt].append((accuracy / float(k),end))
            
    print create_accuracies_markdown_table(data, n_trees)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('--dataset', help="Dataset")

    args = parser.parse_args()

    dataset_filename = args.dataset

    if 'zip' in dataset_filename:
        source_code, labels = extract_and_read_train_files(dataset_filename)
        train('random-forest', source_code, labels)
    else:
        print 'File can only be zip!'