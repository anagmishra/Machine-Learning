import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()

print(data.keys())

dt = pd.DataFrame(data.data)

print(dt)

dt.columns = data.feature_names
dt["heart+disease"] = data.target

print(dt)

print(dt.head())
print(dt.info())

y = dt["heart+disease"]
dt.drop("heart+disease", axis=1)
x = dt

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(y_train.shape)

cls = svm.SVC(kernel = "linear")
#cls = svm.SVC(kernel = "rbf")

cls.fit(x_train, y_train)
y_prediction = cls.predict(x_test)

matrix = confusion_matrix(y_test, y_prediction)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_prediction))