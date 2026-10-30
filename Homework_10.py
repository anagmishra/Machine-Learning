import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("bank-full.csv", sep = ";")

print(data.head())
print(data.info())

numeric_data = data.select_dtypes(include=['number'])

min_max = preprocessing.MinMaxScaler()
min_max.fit(numeric_data)
scaled_data = min_max.transform(numeric_data)

pca = PCA(n_components=3)
pca.fit(scaled_data)

newdt = pca.transform(scaled_data)
print(scaled_data.shape)
print(newdt.shape)

le = preprocessing.LabelEncoder()
y = le.fit_transform(data['y'])

plt.scatter(newdt[:,1], newdt[:,2], c=y)
plt.show()

y = data["y"]
x = data.drop("y", axis=1)

numeric_cols = x.select_dtypes(include=['number']).columns
x_numeric = x[numeric_cols]

x_train, x_test, y_train, y_test = train_test_split(x_numeric, y, test_size=0.2, random_state=1)

min_max = preprocessing.MinMaxScaler()
x_train = min_max.fit_transform(x_train)
x_test = min_max.transform(x_test)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test= le.transform(y_test)

model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

matrix = confusion_matrix(y_test, predictions)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, predictions))
