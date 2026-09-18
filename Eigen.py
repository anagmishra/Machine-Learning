import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()

print(data.keys())
print(data["feature_names"])
dt = pd.DataFrame(data.data)

print(dt)

dt.columns = data.feature_names
dt["breast_cancer"] = data.target

print(dt)

print(dt.head())
print(dt.info())

min_max = preprocessing.MinMaxScaler()
min_max.fit(dt)

scaled_data = min_max.transform(dt)
pca = PCA(n_components=3)
pca.fit(scaled_data)

newdt = pca.transform(scaled_data)
print(scaled_data.shape)
print(newdt.shape)

# plt.scatter(newdt[:,0], newdt[:,1], c=dt["breast_cancer"])
# plt.scatter(newdt[:,1], newdt[:,2], c=dt["breast_cancer"])
plt.scatter(newdt[:,0], newdt[:,2], c=dt["breast_cancer"])
plt.show()

# y = dt["breast_cancer"]
# dt.drop("breast_cancer", axis=1)
# x = dt

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# predictions = model.predict(x_test)
# matrix = confusion_matrix(y_test, predictions)

# sns.heatmap(matrix, annot = True, fmt = 'd')

# plt.title("Confusion Matrix")
# plt.show()

# print("Classification Report:")
# print(classification_report(y_test, predictions))