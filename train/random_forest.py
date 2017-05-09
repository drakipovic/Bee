import pprint

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score


pp = pprint.PrettyPrinter(indent=2)


class RandomForest(object):

    def __init__(self):
        self.rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, criterion="entropy", max_features="log2", oob_score=True)
        
    def train(self, train_features, test_features, train_labels, test_labels):
        print 'Training started...'
        print 'Train feature vector has length of {}'.format(np.array(train_features).shape)
        fs = VarianceThreshold(threshold=(0.6 * (1 - 0.6)))
        train_features = fs.fit_transform(train_features)
        support = fs.get_support(indices=True)

        print 'Feature vector after feature selection has length of {}'.format(np.array(train_features).shape)
        print 'Creating one hot vector from train labels.'
        train_author_indices = self.create_author_indices(train_labels)
        print 'One hot vector created.'
        print 'Fitting random forest...'
        self.rf.fit(train_features, train_author_indices)

        print 'Predicting authors on test set.'
        test_features = np.array(test_features)[:,np.array(support)]
        print 'Test feature vector has length of {}'.format(np.array(test_features).shape)
        predicted_authors_indices = self.rf.predict(test_features)
        print predicted_authors_indices

        test_authors_indices = self.create_author_indices(test_labels)
        
        print test_authors_indices
        pp.pprint(self.seen)

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