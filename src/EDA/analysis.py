import pandas as pd
import numpy as np
import matplotlib as plt
data = pd.read_csv('../../data/heart.csv')
dataFrame = pd.DataFrame(data)

class DataAnalysis:
    def __init__(self , dataframe):
        self.df = dataframe
    def descriptiveStatistics(self):
        print(self.df.shape)
        print(self.df.info())
        print(self.df.describe(include="all"))
    def numericalAnalysis(self):
        numeric = self.df.select_dtypes(include="number")
        print(numeric.describe())
        print(numeric.corr())
    def categoricalAnalysis(self):
        categorical = self.df.select_dtypes(include="object")
        for col in categorical.columns:
            print(col)
            print(categorical[col].value_counts())
            print()
    def missingAnalysis(self):
        missing = self.df.isnull().sum()
        percent = missing / len(self.df) * 100
        print(missing)
        print(percent)
    def duplicateAnalysis(self):
        jumlah = self.df.duplicated().sum()
        print(jumlah)
        if jumlah > 0:
            print(self.df[self.df.duplicated()])
    def uniqueAnalysis(self):
        print(self.df.nunique())
    def cardinalityAnalysis(self):
        kolom_teks = self.df.select_dtypes(include=["number", "string"]).columns
        for kolom in kolom_teks:
            jumlah = self.df[kolom].nunique()
            print(kolom, ":", jumlah, "nilai unik")
            if jumlah > 20:
                print("   -> kardinalitas tinggi, hati-hati saat encoding")
    def distributionAnalysis(self):
        angka = self.df.select_dtypes(include="number")
        print(angka.describe())
        angka.hist(figsize=(10, 8))
        plt.tight_layout()
        plt.show()
    def skewnessAnalysis(self):
        skew = self.df.skew(numeric_only=True)
        for kolom in skew.index:
            nilai = skew[kolom]
            if abs(nilai) < 0.5:
                keterangan = "simetris"
            elif abs(nilai) < 1:
                keterangan = "agak miring"
            else:
                keterangan = "sangat miring"
            print(kolom, ":", round(nilai, 2), "->", keterangan)
    def correlationAnalysis():
        pass
    def targetAnalysis():
        pass
    def classBalanceAnalysis():
        pass
    def dataQualityAnalysis():
        pass
        
init = DataAnalysis(dataframe=dataFrame)
# init.descriptiveStatistics()
# init.numericalAnalysis()
# init.categoricalAnalysis()
init.cardinalityAnalysis()