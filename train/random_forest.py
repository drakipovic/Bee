import pprint

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import accuracy_score


pp = pprint.PrettyPrinter(indent=2)


class RandomForest(object):

    def __init__(self, n_trees=300):
        self.clf = Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'), threshold="mean")),
            ('classification', RandomForestClassifier(n_estimators=n_trees, n_jobs=-1))
        ])

    def train(self, train_features, test_features, train_labels, test_labels):
        print 'Training started...'
        print np.array(train_features).shape
        print np.array(test_features).shape
        print np.array(train_labels).shape
        print np.array(test_labels).shape

        #print 'Train feature vector has length of {}'.format(np.array(train_features).shape)


        #print 'Feature vector after feature selection has length of {}'.format(np.array(train_features).shape)
        #print 'Creating one hot vector from train labels.'
        
        train_author_indices = self.create_author_indices(train_labels)
        #print 'One hot vector created.'
        print 'Fitting random forest...'
        self.clf.fit(train_features, train_author_indices)

        #print 'Predicting authors on test set.'
        #test_features = np.array(test_features)[:,np.array(support)]
        #print 'Test feature vector has length of {}'.format(np.array(test_features).shape)
        predicted_authors_indices = self.clf.predict(test_features)
        #print predicted_authors_indices

        test_authors_indices = self.create_author_indices(test_labels)
        
       # print test_authors_indices
        #pp.pprint(self.seen)

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