from cleaning import DataCleaning
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "cardio_train.csv"
OUTPUT_PATH = DATA_PATH.parent / "cardio_cleaned.csv"

data = pd.read_csv(DATA_PATH, sep=';')

cleaner = DataCleaning(data)
cleaner.dataTypeCorrection()
print('Invalid Values')
cleaner.invalidValueHandling()
print('Missing Data')
cleaner.missingValueHandling()
print('Duplicated Data')
cleaner.duplicateHandling()
cleaner.dataConsistency()
print('Outlier')
cleaner.outlierHandling()
cleaner.finalValidation()
cleaner.saveCleanedData(OUTPUT_PATH)