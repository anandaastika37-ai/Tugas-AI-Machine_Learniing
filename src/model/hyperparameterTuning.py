
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


class HyperparameterTuning:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = self.model
        self.param_distributions = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "class_weight": [None, "balanced"]}
        self.search = None
        self.best_model = None
        self.best_params = None

    def tune(self):
        self.search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_distributions,
            n_iter=20,
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=1,
            verbose=2,
            refit=True
        )
        self.search.fit(self.X_train,self.y_train)

        self.best_model = self.search.best_estimator_
        self.best_params = self.search.best_params_

        print("\nBEST PARAMETERS")
        print(self.best_params)

        print("\nBEST CROSS VALIDATION SCORE")
        print(self.search.best_score_)

        return self.best_model
