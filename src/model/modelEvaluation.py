import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
class ModelEvaluation:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_score = None
        self.results = {}

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        if hasattr(self.model, "predict_proba"):
            self.y_score = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            self.y_score = self.model.decision_function(self.X_test)

        self.results = {
            "Accuracy": accuracy_score(self.y_test, self.y_pred),
            "Precision": precision_score(self.y_test, self.y_pred,zero_division=0),
            "Recall": recall_score(self.y_test, self.y_pred,zero_division=0),
            "F1 Score": f1_score(self.y_test, self.y_pred,zero_division=0),
            "ROC AUC": roc_auc_score(self.y_test, self.y_score) if self.y_score is not None else None
        }
        return pd.DataFrame([self.results])
    
    def confusionMatrix(self):
        return confusion_matrix(self.y_test,self.y_pred,labels=[0, 1])
    
    def classificationReport(self):
        return classification_report(
            self.y_test,
            self.y_pred,
            zero_division=0
        )
