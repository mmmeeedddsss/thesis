from sklearn.model_selection import RandomizedSearchCV


class ClassificationWrapper:
    """Base class for wrapper of classification models"""

    def __init__(self):
        self.classifier = None  # Set on children

    def fit_and_calculate_score(self, X, y, test_X):
        self.classifier = self.classifier.fit(X, y)
        return self.classifier.predict(test_X)

    def optimize_and_fit(self, X, y, test_X):
        optimized_classifier = RandomizedSearchCV(estimator=self.classifier,
                                                  param_distributions=self.parameter_grid,
                                                  n_iter=80, verbose=1,
                                                  random_state=12321, n_jobs=-1)
        optimized_classifier = optimized_classifier.fit(X, y)
        print(optimized_classifier.best_params_)
        return optimized_classifier.best_estimator_.predict(test_X)

    @property
    def parameter_grid(self):
        raise NotImplementedError

