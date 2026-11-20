import numpy as np
import pandas as pd

data = pd.read_csv("games.csv")
print(data.head())
print(data.info())

data['Rating'] = data["Rating"].fillna(data["Rating"].mean())
data["Team"] = data["Team"].fillna(data["Team"].mode()[0])
data["Summary"] = data["Summary"].fillna(data["Summary"].mode()[0])
data = data.drop(columns = ["Unnamed: 0"])
data = data.drop(columns = ["Reviews"])
print(data.head())
print(data.info())