import argparse
from zipfile import ZipFile

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest


def extract_and_read_train_files(train_filename, test_filename):
    print 'Extracting {}'.format(train_filename)

    train_source_code = []
    train_labels = []

    train_zip_file = ZipFile(train_filename, "r")
    for filename in train_zip_file.namelist():
        filename_split = filename.split('_')
        if 'cpp' not in filename_split[-1] and 'cxx' not in filename_split[-1]:
            continue
        
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        train_labels.append(author)

        train_source_code.append(train_zip_file.read(filename))
    
    print 'Extracting {}'.format(test_filename)

    test_source_code = []
    test_labels = []
    

    test_zip_file = ZipFile(test_filename, "r")
    for filename in test_zip_file.namelist():
        filename_split = filename.split('_')
        if 'cpp' not in filename_split[-1] and 'cxx' not in filename_split[-1]:
            continue
        
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        test_labels.append(author)

        test_source_code.append(test_zip_file.read(filename))    
    
    return train_source_code, train_labels, test_source_code, test_labels


def train(ml_algorithm, train_source_code, test_source_code, train_labels, test_labels):
    train_features = FeatureExtractor.get_features(train_source_code, test_source_code)
    test_features = FeatureExtractor.get_features(test_source_code, train_source_code)

    ml_algorithm.train(train_features, test_features, train_labels, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('--train_set', help="Train set")
    parser.add_argument('--test_set', help="Test set")

    args = parser.parse_args()

    train_set_filename = args.train_set
    test_set_filename = args.test_set

    if 'zip' in train_set_filename and 'zip' in test_set_filename:
        train_source_code, train_labels, test_source_code, test_labels = extract_and_read_train_files(train_set_filename,
                                                                                                        test_set_filename)
        train(RandomForest(), train_source_code, test_source_code, train_labels, test_labels)
    else:
        print 'File can only be zip!'