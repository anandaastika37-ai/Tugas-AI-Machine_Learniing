import pandas as pd

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "cardio_train.csv"


class UnderstandingData:
    def __init__(self , dataFrame):
        self.df = dataFrame
    def checkStructure(self):
        rowsAndColumns = self.df.shape
        allData = self.df.size
        structure = self.df.info()
        print(f"Jumlah Baris Dan Kolom : {rowsAndColumns}")
        print(f"Jumlah Semua Data : {allData}")
        print(f"Stuktur Tabel : {structure}")
    def checkTypeData(self):
        typeData = self.df.dtypes
        typeDataNumerical = self.df.select_dtypes(include=["number"])
        countTypeData = self.df.dtypes.value_counts()
        print(f"Tipe Data : {typeData}")
        print(f"Tipe Data Number :{typeDataNumerical}")
        print(f"Jumlah Setiap Type Data : {countTypeData}")
    def checkColumnNames(self):
        collumns = self.df.columns
        collumnsDuplicated = self.df.columns.duplicated()
        listCollums = self.df.columns.tolist()
        print(f"Semua Kolom :{collumns}")
        print(f"Kolom Duplikat :{collumnsDuplicated}")
        print(f"List Nama Kolom : {listCollums}")
    def checkDataSample(self):
        headData = self.df.head(10)
        tailData = self.df.tail(10)
        randomData = self.df.sample(10)
        indexData = self.df.iloc[1]
        describeData = self.df.describe()
        print(f"{headData }")
        print(f"{tailData}")
        print(f"{randomData}")
        print(f"{indexData}")
        print(f"{describeData}")
    def checkIndex(self):
        index = self.df.index
        indexDtype = self.df.index.dtype
        indexUnique = self.df.index.is_unique
        indexMonotonic = self.df.index.is_monotonic_increasing
        indexDuplicated = self.df.index.duplicated()
        indexTerkecil = self.df.index.min()
        indexTerbesar = self.df.index.max()
        print(f"{index}")
        print(f"{indexDtype}")
        print(f"{indexUnique}")
        print(f"{indexMonotonic}")
        print(f"{indexDuplicated}")
        print(f"{indexTerbesar}")
        print(f"{indexTerkecil}")
    def checkMemoryUsage(self):
        memoryUse = self.df.memory_usage()
        memoryUseDetail = self.df.memory_usage(deep=True)
        estiminationMemory = self.df.info(memory_usage="deep")
        print(f"{memoryUse}")
        print(f"{memoryUseDetail}")
        print(f"{estiminationMemory}")
    def checkEmptyData(self):
        emptyData = self.df.empty
        print(f"{emptyData}")
    def checkConstantColumns(self):
        countUniqValue = self.df.nunique()
        listUniqValue = self.df['age'].unique()
        valueCount = self.df.value_counts()
        inputNaN = self.df.dropna=False
        print(f"{countUniqValue}")
        print(f"{listUniqValue}")
        print(f"{valueCount}")
        print(f"{inputNaN}")

        


if __name__ == "__main__":
    dataFrame = pd.read_csv(DATA_PATH, sep=";")
    init = UnderstandingData(dataframe=dataFrame)
    init.cardinalityAnalysis()