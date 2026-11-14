import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../../data/heart_cleaned.csv' , sep=';')
dataframe = pd.DataFrame(data)

y = dataframe['cardio']
x = dataframe.drop(columns=['cardio'])
print(list(x.columns))
print(y.name)