import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class ModelSelection:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(random_state=42)
        }
        self.cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        
    def compareModels(self):
        results = []
        scoring = ["accuracy","precision","recall","f1","roc_auc"]
        for name, model in self.models.items():
            scores = cross_validate(model,self.X_train,self.y_train,cv=self.cv,scoring=scoring,n_jobs=-1)
            results.append({
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1 Score": scores["test_f1"].mean(),
                "ROC AUC": scores["test_roc_auc"].mean()
            })

        self.results = pd.DataFrame(results)

        return self.results.sort_values(by="Recall",ascending=False).reset_index(drop=True)