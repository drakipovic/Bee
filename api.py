import os
from collections import defaultdict

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from bee import basedir, extract_and_read_source_files, train


app = Flask('bee')
cors = CORS(app)


@app.route('/api/deanonymize', methods=['POST'])
def deanonymize():
    train_file = request.files['train_file']
    test_file = request.files['test_file']

    train_filename = secure_filename(train_file.filename)
    test_filename = secure_filename(test_file.filename)

    save_dir = os.path.join(basedir, 'files')

    train_file.save(os.path.join(save_dir, train_filename))
    test_file.save(os.path.join(save_dir, test_filename))

    train_filename = 'files/{}'.format(train_filename)
    test_filename = 'files/{}'.format(test_filename)

    train_source_code, train_labels, test_source_code, test_labels = extract_and_read_source_files(train_filename, test_filename)

    indices, prob = train('random-forest', train_source_code, train_labels, test_source_code, test_labels)

    scores = defaultdict(list)

    for i, idx_list in enumerate(indices):
        sc = []
        for j, idx in enumerate(idx_list):
            sc.append((test_labels[idx], "{0:.2f}".format(prob[i][j])))
        
        scores[i] = sc
    
    return jsonify({"success": True, "scores": scores, "authors": test_labels})


if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')