import pandas as pd

data = pd.read_csv('../data/heart.csv')
dataFrame = pd.DataFrame(data)

class UnderstandingData:
    def __init__(self , dataFrame):
        self.df = dataFrame
    def checkStructure(self):
        print("DATASET INFORMATION")
        self.df.info()
        print(f"\nJumlah baris dan kolom : {self.df.shape}")
        print(f"Jumlah seluruh data   : {self.df.size}")
        print(f"\nDaftar kolom   : {self.df.columns.tolist()}")
        print(f"Jumlah dimensi : {self.df.ndim}")
    def checkData(self):
        print(self.df.head(10)) #10 data teratas
        print(self.df.tail(10)) #10 data paling bawah
        print(self.df.sample(10, random_state=42)) #10 data acak
        print(self.df[["age" , 'thalach']]) # kolom age & thalach
        print(self.df.loc[self.df["age"] > 50]) # filter umur lebih dari 50
        largest = self.df.nlargest(5, "chol").reset_index(drop=True)
        smallest = self.df.nsmallest(5, "chol").reset_index(drop=True)
        print(largest / smallest)
    def dataType(self):
        print("DATA TYPES ANALYSIS")
        print("\n1. Tipe Data Setiap Kolom")
        print(self.df.dtypes)
        print("\n2. Distribusi Tipe Data")
        print(self.df.dtypes.value_counts())
        print("\n3. Kolom Numerikal")
        print(self.df.select_dtypes(include=["number"]).columns.tolist())



init = UnderstandingData(dataFrame)
# print(init.checkData())
# print(init.checkStructure())
# print(init.dataType())
# print(init.missingValue())
# print(init.duplicatedValue())
print(init.StatistikDeskriptif())