import logging

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

logger = logging.getLogger(__name__)

from sklearn.base import clone
import numpy as np

# https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
class OrdinalClassifier:

    def __init__(self):
        self.clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=5, splitter='best', random_state=1)
        self.clfs = {}
        self.unique_class = [1, 2, 3, 4, 5]

    def fit(self, X, y):
        logger.info('Training ordinal classifier')
        for i in tqdm(range(1, 5)):
            binary_y = (y > i).astype(np.uint8)
            clf = clone(self.clf)
            clf.fit(X, binary_y)
            self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)+1
