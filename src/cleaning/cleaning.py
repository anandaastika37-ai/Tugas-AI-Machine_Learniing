import pandas as pd
import numpy as np

data = pd.read_csv("../../data/cardio_train.csv")
dataframe = pd.DataFrame(data)

class DataCleaning:

    def __init__(self, dataframe):
        self.df = dataframe.copy()
    def missingValueHandling(self):
            missing = self.df.isnull().sum()
            print(missing[missing > 0])
            numericalColumns = self.df.select_dtypes(include="number").columns
            for column in numericalColumns:
                self.df[column] = self.df[column].fillna(self.df[column].median())
            return self.df
    def duplicateHandling(self):
            duplicateCount = self.df.duplicated().sum()
            print(f"Total Duplicate: {duplicateCount}")
            self.df = self.df.drop_duplicates()
            return self.df
    def invalidValueHandling(self):
            invalid_rules = {
                "age": lambda x: x <= 0,
                "trestbps": lambda x: x <= 0,
                "chol": lambda x: x <= 0,
                "thalach": lambda x: x <= 0
            }
            for column, condition in invalid_rules.items():
                if column in self.df.columns:
                    invalid = condition(self.df[column])
                    print(f"{column}: {invalid.sum()} invalid values")
                    self.df.loc[invalid, column] = np.nan
            return self.df
    def dataTypeCorrection(self):
        numericalColumns = ["age", "trestbps", "chol", "thalach"]
        for column in numericalColumns:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column],errors="coerce")
        print(self.df.dtypes)
        return self.df
    def outlierHandling(self):
            numerical_columns = self.df.select_dtypes(include="number").columns
            for column in numerical_columns:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier = (
                    (self.df[column] < lower) |
                    (self.df[column] > upper)
                )

                print(f"{column}: {outlier.sum()} outliers")
            return self.df
    def dataConsistency(self):
        categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        for column in categorical_columns:
            self.df[column] = (self.df[column].astype("string").str.strip().str.lower())
        return self.df
    def finalValidation(self):
            print("FINAL DATA VALIDATION")
            print("\nDataset Shape:")
            print(self.df.shape)
            print("\nMissing Values:")
            print(self.df.isnull().sum())
            print("\nDuplicate Values:")
            print(self.df.duplicated().sum())
            print("\nData Types:")
            print(self.df.dtypes)
            print("\nStatistical Description:")
            print(self.df.describe())
            return self.df
    def saveCleanedData(self, path="../../data/heart_cleaned.csv"):
        self.df.to_csv(path, index=False)
        print(f"Cleaned dataset berhasil disimpan: {path}")
        return self.df
