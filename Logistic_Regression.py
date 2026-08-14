import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("titanic.csv")

print(data.head())
print(data.info())

le = preprocessing.LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

print(data.head())
print(data.info())

x = data[["Sex", "Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"]]
y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(y_train.shape)

model = LogisticRegression()
model.fit(x_train, y_train)

y_predictions = model.predict(x_test)

matrix = confusion_matrix(y_test, y_predictions)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_predictions))