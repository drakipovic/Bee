import os
import shutil
import random
from zipfile import ZipFile
from collections import defaultdict

from bee import basedir


def create_bee_readable_zip_file(filename):
    zip_file = ZipFile(filename, "r")

    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    
    os.makedirs('dataset')

    authors = defaultdict(int)
    possible = {}
    for filename in sorted(zip_file.namelist()):
        if len(zip_file.read(filename).split('\n')) < 20: 
            continue

        f_split = filename.split('/')

        name = f_split[0]
        f_split = f_split[1].split('.')
        lang = f_split[1]
        task_name = f_split[0].split('_')[1]

        if 'cpp' in lang or 'cxx' in lang:
            authors[name] += 1
        
        possible[name] = 11
    

    file_list = zip_file.namelist()
    random.shuffle(file_list)
    
    for filename in file_list:
        if len(zip_file.read(filename).split('\n')) < 20: 
            continue
        f_split = filename.split('/')

        name = f_split[0]
        lang = f_split[-1]

        if authors[name] >= 11 and ('cpp' in lang or 'cxx' in lang) and possible[name] > 0:
            new_filename = name + '_' + filename.split('/')[1]
            f = open('dataset/' + new_filename, 'w')
            f.write(zip_file.read(filename))
            possible[name] -= 1
    
    if os.path.exists('test_files'):
        shutil.rmtree('test_files')

    os.makedirs('test_files')

    files = os.listdir('dataset')
    random.shuffle(files)
    
    visited = []
    for filename in files:
        f_split = filename.split('_')

        name = f_split[0] + '_' + f_split[1]
        if name in visited:
            continue
        
        visited.append(name)
        shutil.move(basedir + '/dataset/' + filename, basedir + '/test_files/' + filename)

        

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
    