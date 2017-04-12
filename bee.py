import argparse
from zipfile import ZipFile


def read_and_extract_zip(filename):
    print 'Extracting {}'.format(filename)

    zip_file = ZipFile(filename, "r")
    for filename in zip_file.namelist():
        print zip_file.read(filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deanonymization bee')

    parser.add_argument('-f', '--file', help="input file")

    args = parser.parse_args()

    filename = args.file
    if 'zip' in filename:
        read_and_extract_zip(filename)
    else:
        print 'File can only be zip!'