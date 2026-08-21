import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")

print(data.head())
print(data.info())

x = data[["petal.length", "petal.width", "sepal.length", "sepal.width"]]
y = data["variety"]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

min_max = preprocessing.MinMaxScaler()
x_train = min_max.fit_transform(x_train)
x_test = min_max.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
matrix = confusion_matrix(y_test, predictions)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, predictions))