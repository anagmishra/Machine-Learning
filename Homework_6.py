from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("bank-full.csv", sep = ";")

print(data.head())
print(data.info())

x = data[["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]]
y = data["y"]

le = preprocessing.LabelEncoder()
oe = preprocessing.OrdinalEncoder()

x = oe.fit_transform(x)
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(y_train.shape)

#model = DecisionTreeClassifier(criterion="gini", random_state=1)
model = DecisionTreeClassifier(criterion="entropy", random_state=1)
model.fit(x_train, y_train)

y_predictions = model.predict(x_test)

matrix = confusion_matrix(y_test, y_predictions)

sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_predictions))

x = data[["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]]

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=x.columns,
    class_names=le.inverse_transform(np.unique(y)),
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("bank_decision_tree")
graph.view()