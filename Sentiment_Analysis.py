import pandas as pd
from sklearn.model_selection import train_test_split
import nltk


data = pd.read_csv("train.txt", sep = ";", names = ["text", "label"])

print(data.head())
print(data.info())

data["label"] = data["label"].replace({"sadness": 1, "anger":1, "fear":1, "joy":0, "love":0, "surprise":0})

print(data.head())

x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

wne = WordNetLemmatizer()

def TransformWords(data):
    corpus = []
    for i in data:
        newi = re.sub("[^a-zA-Z]", " ", i)
        newi = newi.lower()
        newi = newi.split()
        list1 = []
        for word in newi:
            if word not in stopwords.words("english"):
                list1.append(wne.lemmatize(word))
        corpus.append("".join(list1))
    return corpus

x_train_clean = TransformWords(x_train)
x_test_clean = TransformWords(x_test)

print(x_test_clean[0:5])