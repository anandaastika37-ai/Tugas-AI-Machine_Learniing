import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "cardio_train.csv"

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
        isNan = self.df.isna().sum()
        percent = missing / len(self.df) * 100
        print(missing)
        print(isNan)
        print(percent)
    def duplicateAnalysis(self):
        jumlah = self.df.duplicated().sum()
        print(jumlah)
        if jumlah > 0:
            print(self.df[self.df.duplicated()])
    def uniqueAnalysis(self):
        result = {}
        totalRows = len(self.df)
        for collum in self.df.columns:
            series = self.df[collum]
            uniqeValue = series.unique()
            nuniqeValue = series.nunique()
            frequency = series.value_counts(dropna=False)
            percentage = (series.value_counts(normalize=True , dropna=False) * 100)
            mode = series.mode().tolist()
            if totalRows > 0:
                uniquePercen = (nuniqeValue / totalRows * 100)
            else : 
                uniquePercen = 0
            result[collum] = {
                "UniqCount" : nuniqeValue,
                "percenUniq" : uniquePercen,
                "missngCount" : series.isna().sum(),
                "mode" : mode,
                "uniqueValue" : uniqeValue.tolist()
            }

            print(f"{6*'='}", "Unique Analysis", (f"{6*'='}"))
            print(f"\n Kolom = {collum}")
            print(f"Unique Count : {nuniqeValue}")
            print(f"Unique Percentage : {uniquePercen:.2f}%")
            print(f"Missing Count: {series.isna().sum()}")
            print(f"Mode : {mode}")

            print("\nUnique Values:")
            print(uniqeValue)
            print("\nFrequency Distribution:")

            print(pd.DataFrame({
                'frequency': frequency,
                'percentage': percentage.round(2)
            }))
        return pd.DataFrame.from_dict(result , orient="index")
    def cardinalityAnalysis(self):
        result = {}
        totalRows = len(self.df)
        for collum in self.df.columns:
            series = self.df[collum]
            cardinality = series.nunique()
            if totalRows:
                cardinalityPercen = cardinality / totalRows * 100
            else:
                cardinalityPercen = 0
            missing = series.isna().sum()

            match cardinality:
                case 0:
                    category = "Empty Column"
                case 1:
                    category = "Constant"
                case 2:
                    category = "Binary"
                case n if n <= 10:
                    category = "Low Cardinality"
                case _:
                    category = "High Cardinality"
            result[collum] = {
                    'cardinality': cardinality,
                    'cardinalityPercentage': round(
                        cardinalityPercen, 2
                    ),
                    'missing_count': missing,
                    'classification': category
            }
            print(f"{6*'='}", "Cardinality Analysis", (f"{6*'='}"))
            print(f"Kolom = {collum}")
            print(f"Cardinality: {cardinality}")
            print(
                f"Cardinality Percentage: "
                f"{cardinalityPercen:.2f}%"
            )
            print(f"Missing Values: {missing}")
            print(f"Classification: {category}")

            
            report = pd.DataFrame.from_dict(
                result,
                orient='index'
            )
        return report

    def distributionAnalysis(self):
        for collom in self.df.columns:
            series = self.df[collom].dropna()

            print(f"{6*'='}", "Distribution Analysis", (f"{6*'='}"))
            print(f"\nKolom: {collom}")
            print(f"Jumlah data: {len(series)}")
            print("\nDistribusi Frekuensi:")
            print(series.value_counts().sort_index())
            print("\nDistribusi Persentase:")
            print((series.value_counts(normalize=True) * 100).sort_index().round(2))
            # Statistik distribusi untuk data numerik
            if pd.api.types.is_numeric_dtype(series):
                print("\nStatistik Distribusi:")
                print(f"Minimum : {series.min()}")
                print(f"Maximum : {series.max()}")
                print(f"Mean : {series.mean():.2f}")
                print(f"Median : {series.median():.2f}")
                print(f"Std : {series.std():.2f}")
    def skewnessAnalysis(self):
        numericDataframe = self.df.select_dtypes(include="number")
        skewness = numericDataframe.skew()
        results = []
        for col, value in skewness.items():
            if pd.isna(value):
                category = "Undefined (constant column)"
            elif -0.5 <= value <= 0.5:
                category = "Relatively Symmetrical"
            elif 0.5 < value <= 1:
                category = "Moderately Right-Skewed"
            elif value > 1:
                category = "Highly Right-Skewed"
            elif value >= -1:
                category = "Moderately Left-Skewed"
            else:
                category = "Highly Left-Skewed"
            results.append({
                "column": col,
                "skewness": round(value, 3),
                "distribution": category
            })

        resultData = pd.DataFrame(results)
        print(resultData)
        return resultData
    def correlationAnalysis(self , target="cardio"):
        numericDatafream = self.df.select_dtypes(include="number")
        correlation = numericDatafream.corr(method="pearson")
        print("\nCorrelation Matrix:")
        print(correlation.round(2))
        if target in correlation.columns:
            target_corr = correlation[target].drop(target)
            target_corr = target_corr.reindex(
                target_corr.abs().sort_values(ascending=False).index
            )
            print(f"\nCorrelation terhadap {target}:")
            print(target_corr.round(3))
        return correlation
    def targetAnalysis(self , target="cardio"):
            if target not in self.df.columns:
                print(f"Kolom target '{target}' tidak ditemukan!")
                return None
            series = self.df[target]
            totalData = len(series)
            missing = series.isna().sum()
            classCount = series.value_counts(dropna=False)
            classPercentage = (
                series.value_counts(normalize=True, dropna=False) * 100
            )
            report = pd.DataFrame({
                "jumlah": classCount,
                "persentase": classPercentage.round(2)
            })

            print(f"Total data : {totalData}")
            print(f"Missing target : {missing}")
            print(f"Jumlah kelas : {series.nunique()}")
            print("\nDistribusi Target:")
            print(report)
            validCount = series.value_counts()
            if len(validCount) >= 2 and validCount.iloc[-1] > 0:
                imbalanceRatio = validCount.iloc[0] / validCount.iloc[-1]
                print(f"\nImbalance Ratio  : {imbalanceRatio:.2f}:1")
            return report
    
    def classBalanceAnalysis(self):
            target = 'cardio'
            classCount = self.df[target].value_counts()
            classPercentage = self.df[target].value_counts(normalize=True) * 100
            classBalance = pd.DataFrame({
                'Count': classCount,
                'Percentage': classPercentage
            })
            print("\nDistribusi kelas:")
            print(classBalance)
            majorityClass = classCount.idxmax()
            minorityClass = classCount.idxmin()
            majorityCount = classCount.max()
            minorityCount = classCount.min()
            print(f"Kelas mayoritas: {majorityClass}")
            print(f"Jumlah mayoritas: {majorityCount}")
            print(f"Kelas minoritas: {minorityClass}")
            print(f"Jumlah minoritas: {minorityCount}")
            imbalanceRatio = majorityCount / minorityCount
            print(f"Imbalance Ratio: {imbalanceRatio:.2f}:1")
            if imbalanceRatio <= 1.5:
                print("Status: Relatif seimbang")
            elif imbalanceRatio <= 3:
                print("Status: Moderately imbalanced")
            else:
                print("Status: Highly imbalanced")
    def dataQualityAnalysis(self):
            missing = self.df.isnull().sum()
            totalMissing = missing.sum()
            duplicate = self.df.duplicated().sum()
            dataTypes = self.df.dtypes
            constantColumns = [
                col for col in self.df.columns
                if self.df[col].nunique(dropna=False) <= 1
            ]
            numericData = self.df.select_dtypes(include='number')
            infiniteValues = np.isinf(numericData).sum()
            totalInfinite = infiniteValues.sum()
            uniqueValues = self.df.nunique()
            missingPercentage = (
                self.df.isnull().mean() * 100
            )
            qualityReport = pd.DataFrame({
                'Data Type': dataTypes,
                'Unique Values': uniqueValues,
                'Missing Values': missing,
                'Missing (%)': missingPercentage,
            })

            print("\n--- Data Quality Report ---")
            print(qualityReport)
            print("\n--- Overall Summary ---")
            print(f"Total Rows : {self.df.shape[0]}")
            print(f"Total Columns : {self.df.shape[1]}")
            print(f"Missing Values : {totalMissing}")
            print(f"Duplicate Rows : {duplicate}")
            print(f"Infinite Values : {totalInfinite}")
            print(f"Constant Columns : {len(constantColumns)}")

            print("\n--- Constant Columns ---")
            print(constantColumns)

            print("\n--- Data Quality Status ---")

            if (
                totalMissing == 0
                and duplicate == 0
                and totalInfinite == 0
                and len(constantColumns) == 0
            ):
                print("Dataset memiliki kualitas dasar yang baik.")
            else:
                print("Dataset membutuhkan pemeriksaan dan cleaning.")
        
if __name__ == "__main__":
    dataFrame = pd.read_csv(DATA_PATH, sep=";")
    init = DataAnalysis(dataframe=dataFrame)
    init.cardinalityAnalysis()