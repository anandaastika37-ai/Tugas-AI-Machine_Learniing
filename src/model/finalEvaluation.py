
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve)


class FinalModelEvaluation:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        self.y_pred = None
        self.y_score = None
        self.results = {}

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        if hasattr(self.model, "predict_proba"):
            self.y_score = self.model.predict_proba(
                self.X_test
            )[:, 1]
        else:
            self.y_score = self.model.decision_function(self.X_test)
        return self.y_pred

    def calculateMetrics(self):
        if self.y_pred is None:
            self.predict()
        self.results = {
            "Accuracy": accuracy_score(
                self.y_test, self.y_pred
            ),
            "Precision": precision_score(
                self.y_test, self.y_pred,
                zero_division=0
            ),
            "Recall": recall_score(
                self.y_test, self.y_pred,
                zero_division=0
            ),
            "F1 Score": f1_score(
                self.y_test, self.y_pred,
                zero_division=0
            ),
            "ROC AUC": roc_auc_score(
                self.y_test, self.y_score
            )
        }

        return self.results

    def classificationReport(self):
        if self.y_pred is None:
            self.predict()

        print("\nCLASSIFICATION REPORT")
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                zero_division=0
            )
        )

    def confusionMatrix(self):

        if self.y_pred is None:
            self.predict()

        cm = confusion_matrix(self.y_test,self.y_pred)

        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=[0, 1],yticklabels=[0, 1])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.show()

        return cm

    def rocCurve(self):

        if self.y_pred is None:
            self.predict()

        fpr, tpr, thresholds = roc_curve(self.y_test,self.y_score)
        auc = roc_auc_score(self.y_test,self.y_score)

        plt.plot(fpr, tpr,label=f"ROC Curve (AUC = {auc:.4f})")

        plt.plot([0, 1], [0, 1],linestyle="--",label="Random Classifier")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return auc

    def evaluate(self):
        self.predict()
        metrics = self.calculateMetrics()
        print("\nFINAL MODEL EVALUATION")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        self.classificationReport()
        self.confusionMatrix()
        self.rocCurve()

        return metrics
