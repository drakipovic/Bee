import numpy as np
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score


class SVM(object):

    def __init__(self, variance_threshold=0.08):
        self.svm = svm.SVC()
        self.variance_threshold = variance_threshold

    def train(self, train_features, test_features, train_labels, test_labels):
        fs = VarianceThreshold(threshold=self.variance_threshold)
        train_features = fs.fit_transform(train_features)
        
        support = fs.get_support(indices=True)

        self.svm.fit(train_features, train_labels)

        test_features = np.array(test_features)[:,np.array(support)]
        predicted_authors = self.svm.predict(test_features)

        accuracy = accuracy_score(test_labels, predicted_authors)
        return accuracy