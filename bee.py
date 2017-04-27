import argparse
from zipfile import ZipFile

from feature_extraction import FeatureExtractor
from train.random_forest import RandomForest

def read_and_extract_zip(filename):
    print 'Extracting {}'.format(filename)

    train_features = []
    correct_class = []

    zip_file = ZipFile(filename, "r")
    for filename in zip_file.namelist():
        filename_split = filename.split('_')
        author = "{} {}".format(filename_split[0].encode('utf-8'), filename_split[1].encode('utf-8'))
        prog_language = filename_split[-1].split('.')[1]

        train_features.append(FeatureExtractor.get_features('cpp', zip_file.read(filename))[0])
        correct_class.append(author)
        
    
    rf = RandomForest()
    rf.train(train_features, correct_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('-f', '--file', help="input file")

    args = parser.parse_args()

    filename = args.file
    if 'zip' in filename:
        read_and_extract_zip(filename)
    else:
        print 'File can only be zip!'