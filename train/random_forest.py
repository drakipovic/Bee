import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


class RandomForest(object):

    def __init__(self, n_trees=500, threshold=0, svc=True):
        if svc:
            self.clf = Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'), threshold="mean")),
                ('classification', RandomForestClassifier(n_estimators=n_trees, 
                                                            n_jobs=-1, 
                                                            oob_score=True, 
                                                            min_samples_leaf=1))
            ])
        else:
            self.clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
            self.threshold = threshold
        
        self.svc = svc

    def fit(self, train_features, train_labels, name):
        train_author_indices = self.create_author_indices(train_labels)

        self.clf.fit(train_features, train_author_indices)

        joblib.dump(self.clf, 'trained_classificators/{}.pkl'.format(name))
        
    def predict(self, test_features, name):
        self.clf = joblib.load('trained_classificators/{}.pkl'.format(name))

        authors_prob = self.clf.predict_proba(test_features)

        k = 3
        prob_ind = np.argpartition(authors_prob, -k, axis=1)[:,range(-k, 0)]
        highest_scores = authors_prob[np.array([[i]*k for i in range(prob_ind.shape[0])]), prob_ind]

        return prob_ind, highest_scores

    def fit_and_predict(self, train_features, test_features, train_labels, test_labels):
        train_author_indices = self.create_author_indices(train_labels)

        if not self.svc:
            fs = VarianceThreshold(threshold=self.threshold)
            train_features = fs.fit_transform(train_features)
            support = fs.get_support(indices=True)

        print np.array(train_features).shape
        self.clf.fit(train_features, train_author_indices)

        if not self.svc:
            test_features = np.array(test_features)[:,np.array(support)]

        predicted_authors_indices = self.clf.predict(test_features)
        print self.clf.named_steps['feature_selection'].get_support().shape
        test_authors_indices = self.create_author_indices(test_labels)
        
        accuracy = accuracy_score(test_authors_indices, predicted_authors_indices)
        
        return accuracy

    def create_author_indices(self, authors):
        self.seen = {}
        index = 0
        self.indices = []

        for author in authors:
            if self.seen.get(author) != None:
                self.indices.append(self.seen.get(author))
            else:
                self.indices.append(index)
                self.seen[author] = index
                index += 1
        
        return self.indices