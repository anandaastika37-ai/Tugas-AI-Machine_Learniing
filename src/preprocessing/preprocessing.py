import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Preprocessing:
    def __init__(self, X_train, X_test):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()

        self.preprocessor = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.feature_names = None

    def buildPipeline(self):
        numericFeatures = self.X_train.select_dtypes(include=np.number).columns.tolist()
        categoricalFeatures = self.X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numericPipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
        categoricalPipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('numeric', numericPipeline, numericFeatures),
            ('categorical', categoricalPipeline, categoricalFeatures)
        ])
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)

        self.feature_names = self.preprocessor.get_feature_names_out()

        return self

    def getProcessedData(self):
        return self.X_train_processed, self.X_test_processed