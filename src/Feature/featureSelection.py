import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureSelection:
    def __init__(self, X_train, X_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

    def selectFeatures(self, y_train, k=10):
        k = min(k, self.X_train.shape[1])
        self.selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=k)
        self.X_train_selected = self.selector.fit_transform(self.X_train, y_train)
        self.X_test_selected = self.selector.transform(self.X_test)
        self.selected_features = self.feature_names[self.selector.get_support()]
        self.feature_scores = pd.DataFrame({'feature': self.feature_names,'score': self.selector.scores_}).sort_values('score', ascending=False)
        return self
    
    def getSelectedFeatures(self):
        return self.X_train_selected, self.X_test_selected

    def getFeatureScores(self):
        return self.feature_scores

    def getSelectedFeatureNames(self):
        return self.selected_features
