import argparse
from zipfile import ZipFile

import numpy as np

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest


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


def train(ml_algorithm, source_code, labels):
    source_code = np.array(source_code)
    labels = np.array(labels)

    k = 10
    dataset_size = len(source_code)
    fold_size = dataset_size / k
    accuracy = 0

    for i in range(k):
        test_indices = np.array(range(i, len(source_code), fold_size))
        
        train_indices = []
        for sci in range(len(source_code)):
            if sci not in test_indices:
                train_indices.append(sci)
        
        train_features, test_features = FeatureExtractor.get_features(source_code[train_indices], source_code[test_indices])
        score = ml_algorithm.train(train_features, test_features, labels[train_indices], labels[test_indices])
        print 'Score after {}th fold: {}'.format(i, score)
        accuracy += score
    
    print 'Final score: {}'.format(accuracy / float(k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('--dataset', help="Dataset")

    args = parser.parse_args()

    dataset_filename = args.dataset

    if 'zip' in dataset_filename:
        source_code, labels = extract_and_read_train_files(dataset_filename)
        train(RandomForest(), source_code, labels)
    else:
        print 'File can only be zip!'