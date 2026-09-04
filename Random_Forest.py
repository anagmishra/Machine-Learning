from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv")

print(data.head())
print(data.info())

#x = data[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences", "G1", "G2"]]
#x = data[["school","sex","age","address","reason","traveltime","paid","activities", "internet","romantic","freetime","goout","Dalc","Walc","health"]]
#x = data[["famsize","Pstatus","Medu","Fedu","Mjob","Fjob","guardian","famsup","famrel","schoolsup","nursery","higher"]]
x = data[["studytime","failures","absences", "G1", "G2"]]
y = data["G3"]

le = preprocessing.LabelEncoder()
oe = preprocessing.OrdinalEncoder()

x = oe.fit_transform(x)
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(y_train.shape)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_predictions = model.predict(x_test)

matrix = confusion_matrix(y_test, y_predictions)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_predictions))
