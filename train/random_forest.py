import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self):
        self.rf = RandomForestClassifier(n_jobs=-1, n_estimators=5)
        
    def train(self, source_code_train_set, authors):
        one_hot = self.create_one_hot(authors)
        train_data = np.array(source_code_train_set)

        self.rf.fit(source_code_train_set, one_hot)

        predicted_authors_oh = self.rf.predict(source_code_train_set)
        predicted_authors = np.argmax(predicted_authors_oh, 1)

        correct = (self.indices == predicted_authors).sum()

        print correct / float(len(self.indices))

    def create_one_hot(self, authors):
        seen = {}
        index = 0
        self.indices = []

        for author in authors:
            if seen.get(author):
                self.indices.append(seen.get(author))
            else:
                self.indices.append(index)
                seen[author] = index
                index += 1
        
        n_labels = np.max(self.indices)

        one_hot = np.eye(n_labels+1)[self.indices]

        return one_hot