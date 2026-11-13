import pandas as pd
import numpy as np

data = pd.read_csv('../data/heart.csv')
dataFrame = pd.DataFrame(data)

class DataAnalysis:
    def __init__(self , dataframe):
        self.df = dataframe
    def missingValue(self):
        missing_count = self.df.isnull().sum()
        totalMissing , ukuran = missing_count.sum() , self.df.size
        print(f"{totalMissing / ukuran * 100:.2f}%")
        print(f"{len(self.df.dropna())} dari  {len(self.df)}")
        print(self.df[self.df.isnull().any(axis=1)]) 
        print(pd.DataFrame({"jumlah": missing_count,"persen": (missing_count / len(self.df) * 100).round(2)}).sort_values("persen", ascending=False))
    def duplicatedValue(self):
        print(self.df.duplicated().sum())
        duplicate_count = self.df.duplicated().sum()
        duplicate_mask = self.df.duplicated(keep=False)
        duplicate_rows = self.df[duplicate_mask]
        unique_count = self.df.drop_duplicates()
        print(unique_count.index.tolist())
        print(duplicate_rows.index.tolist())
        duplicates = self.df[self.df.duplicated(keep=False)]
        print(duplicates.sort_values(by=self.df.columns.tolist()))
        print(self.df.duplicated().sum() / len(self.df) * 100)
        print(duplicate_rows.sort_values(by=self.df.columns.tolist()))
        print(unique_count["target"].value_counts())
        print(self.df['target'].value_counts())
    def invalidValue(self):
        print(self.df.describe())
    def categoricalAnalysis():
        pass


