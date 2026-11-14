from cleaning import DataCleaning
import pandas as pd

data = pd.read_csv("../../data/cardio_train.csv")


cleaner = DataCleaning(data)
cleaner.dataTypeCorrection()
cleaner.invalidValueHandling()
cleaner.missingValueHandling()
cleaner.duplicateHandling()
cleaner.dataConsistency()
cleaner.outlierHandling()
cleaner.finalValidation()
cleaner.saveCleanedData()