import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self):
        self.rf = RandomForestClassifier(n_jobs=-1, n_estimators=5)
        
    def train(self, train_features, test_features, train_labels, test_labels):
        print 'Training started...'
        print 'Feature vector for one source code has length of {}'.format(np.array(train_features[0]).shape)
        print 'Creating one hot vector.'
        one_hot = self.create_one_hot(train_labels)
        print 'One hot vector created.'

        print 'Fitting random forest...'
        self.rf.fit(train_features, one_hot)

        print 'Predicting authors on test set.'
        predicted_authors_oh = self.rf.predict(test_features)
        test_oh = self.create_one_hot(test_labels)
        predicted_authors = np.argmax(predicted_authors_oh, 1)

        correct = (self.indices == predicted_authors).sum()

        print 'Score: {}'.format(correct / float(len(self.indices)))

    def create_one_hot(self, authors):
        seen = {}
        index = 0
        self.indices = []

        for author in authors:
            if seen.get(author) != None:
                self.indices.append(seen.get(author))
            else:
                self.indices.append(index)
                seen[author] = index
                index += 1

        n_labels = np.max(self.indices)

        one_hot = np.eye(n_labels+1)[self.indices]

        return one_hot