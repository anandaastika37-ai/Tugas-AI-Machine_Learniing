import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaning:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.report = {}

    def missingValueHandling(self):
            missing = self.df.isnull().sum()
            missing = missing[missing > 0]
            intColunm = ["age", "height", "ap_hi", "ap_lo"]
            self.report["missing"] = missing.to_dict()
            print(missing if not missing.empty else "Tidak ada missing value")

            for column in self.df.select_dtypes(include="number").columns:
                self.df[column] = self.df[column].fillna(self.df[column].median())

            for column in intColunm:
                if column in self.df.columns:
                    self.df[column] = self.df[column].round().astype(int)
            return self.df
    
    def duplicateHandling(self):
                    subset = [c for c in self.df.columns if c != "id"]
                    duplicateCount = int(self.df.duplicated(subset=subset).sum())
                    self.report["duplicates_removed"] = duplicateCount
                    print(f"Total Duplicate: {duplicateCount}")
                    self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
                    return self.df
    
    def invalidValueHandling(self):
            invalid_rules = {
            "height": lambda x: (x < 120) | (x > 220),
            "weight": lambda x: (x <= 30) | (x > 200),
            "ap_hi": lambda x: (x <= 70) | (x > 250),
            "ap_lo": lambda x: (x <= 40) | (x > 150),
            }
            report = {}
            for column, condition in invalid_rules.items():
                if column in self.df.columns:
                    invalid = condition(self.df[column])
                    report[column] = int(invalid.sum())
                    print(f"{column}: {invalid.sum()} invalid values")
                    self.df.loc[invalid, column] = np.nan

            if {"ap_hi", "ap_lo"}.issubset(self.df.columns):
                reversed_bp = self.df["ap_lo"] >= self.df["ap_hi"]
                report["ap_lo >= ap_hi"] = int(reversed_bp.sum())
                print(f"ap_lo >= ap_hi: {reversed_bp.sum()} invalid rows")
                self.df.loc[reversed_bp, ["ap_hi", "ap_lo"]] = np.nan

            self.report["invalid"] = report
            kolom = ["height", "weight", "ap_hi", "ap_lo"]
            print("Jumlah baris yang punya nilai invalid:", self.df[kolom].isna().any(axis=1).sum())
            return self.df
        
    def dataTypeCorrection(self):
        numericalColumns = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
        for column in numericalColumns:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column],errors="coerce")
        print(self.df.dtypes)
        return self.df
    
    def outlierHandling(self , action ='report'):
            numerColums = ["age", "height", "weight", "ap_hi", "ap_lo"]
            report = {}
            for column in numerColums:
                if column not in self.df.columns:
                    continue
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier = (self.df[column] < lower) | (self.df[column] > upper)
                report[column] = int(outlier.sum())
                print(f"{column}: {outlier.sum()} outliers")

                if action == "clip":
                    self.df[column] = self.df[column].clip(lower, upper)
                    if column in self.INTEGER_COLUMNS:
                        self.df[column] = self.df[column].round().astype(int)

            self.report["outliers"] = report
            self.report["outlier_action"] = action
            return self.df
    
    def dataConsistency(self):
        categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        for column in categorical_columns:
            self.df[column] = (self.df[column].astype("string").str.strip().str.lower())
        return self.df
    
    def finalValidation(self):
            print("=" * 40)
            print("RINGKASAN PROSES CLEANING")
            print("=" * 40)
            for step, result in self.report.items():
                print(f"{step}: {result}")

            print("\n" + "=" * 40)
            print("KONDISI DATA AKHIR")
            print("=" * 40)
            print("\nDataset Shape:")
            print(self.df.shape)
            print("\nMissing Values:")
            print(self.df.isnull().sum())
            print("\nDuplicate Values (tanpa id):")
            print(self.df.duplicated(subset=[c for c in self.df.columns if c != "id"]).sum())
            print("\nData Types:")
            print(self.df.dtypes)
            print("\nStatistical Description:")
            print(self.df.describe())
            return self.df

    def saveCleanedData(self, path="../../data/heart_cleaned.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        print(f"Cleaned dataset berhasil disimpan: {path}")
        return self.df
