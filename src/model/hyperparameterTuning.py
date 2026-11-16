from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class HyperparameterTuning:
    def __init__(self, model, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.param_distributions = self.getParameters()
        self.search = None
        self.best_model = None
        self.best_params = None

    def getParameters(self):
        if isinstance(self.model, SVC):
            return {
                "C": [0.1, 1, 10, 100],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "class_weight": [None, "balanced"]
            }
        elif isinstance(self.model, RandomForestClassifier):
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced", "balanced_subsample"]
            }
        elif isinstance(self.model, LogisticRegression):
            return {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"]
            }
        else:
            raise ValueError(f"Model {type(self.model).__name__} belum didukung.")
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
            refit=True)
        self.search.fit(self.X_train, self.y_train)
        self.best_model = self.search.best_estimator_
        self.best_params = self.search.best_params_
        print("\nBEST PARAMETERS")
        print(self.best_params)
        print("\nBEST CROSS VALIDATION SCORE")
        print(self.search.best_score_)

        return self.best_model
