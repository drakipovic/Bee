import os
from zipfile import ZipFile
from collections import defaultdict

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from bee import basedir, extract_and_read_source_files, train_and_predict, fit, get_probs


save_dir = os.path.join(basedir, 'files')


app = Flask('bee')
cors = CORS(app)


@app.route('/api/train', methods=['POST'])
def train():
    name = request.form['name']

    train_file = request.files['train_file']
    train_filename = secure_filename(train_file.filename)

    train_file.save(os.path.join(save_dir, train_filename))

    train_source_code, train_labels = extract_and_read_source_files(train_filename)

    fit('random-forest', train_source_code, train_labels, name)

    return jsonify({"success": True})


@app.route('/api/predict', methods=['POST'])
def predict():
    name = request.form['name']
    test_file = request.files['test_file']

    train_filename = '{}.zip'.format(name)
    test_filename = secure_filename(test_file.filename)
    test_file.save(os.path.join(save_dir, test_filename))

    train_source_code, _, test_source_code, test_labels = extract_and_read_source_files(train_filename, test_filename)

    prob_ind, prob = get_probs('random-forest', train_source_code, test_source_code, name)
    
    scores = defaultdict(list)

    for i, idx_list in enumerate(prob_ind):
        sc = []
        for j, idx in enumerate(idx_list):
            sc.append((test_labels[idx], "{0:.2f}".format(prob[i][j])))
        
        scores[i] = sc
    
    zip_file = ZipFile(os.path.join(save_dir, test_filename), "r")
    code_names = sorted(zip_file.namelist())
    
    return jsonify({"success": True, "scores": scores, "code_names": code_names})


@app.route('/api/deanonymize', methods=['POST'])
def deanonymize():
    train_file = request.files['train_file']
    test_file = request.files['test_file']

    train_filename = secure_filename(train_file.filename)
    test_filename = secure_filename(test_file.filename)

    train_file.save(os.path.join(save_dir, train_filename))
    test_file.save(os.path.join(save_dir, test_filename))

    train_source_code, train_labels, test_source_code, test_labels = extract_and_read_source_files(train_filename, test_filename)

    indices, prob = train_and_predict('random-forest', train_source_code, train_labels, test_source_code, test_labels)

    scores = defaultdict(list)

    for i, idx_list in enumerate(indices):
        sc = []
        for j, idx in enumerate(idx_list):
            sc.append((test_labels[idx], "{0:.2f}".format(prob[i][j])))
        
        scores[i] = sc
    
    return jsonify({"success": True, "scores": scores, "authors": test_labels})


@app.route('/api/trained-files')
def get_trained_files():
    files = os.listdir('trained_classificators')
    files = [file.split('.')[0] for file in files]

    return jsonify({"files": files})


if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')