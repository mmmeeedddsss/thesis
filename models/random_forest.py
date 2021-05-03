import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.classification_wrapper import ClassificationWrapper


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
class RandomForestWrapper(ClassificationWrapper):
    def __init__(self):
        print('Random Forest')
        self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=12321)

    @property
    def parameter_grid(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=20, stop=500, num=30)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 100, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        return {
            'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
            'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap
        }
