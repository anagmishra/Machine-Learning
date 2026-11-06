import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from sklearn.feature_extraction.text import CountVectorizer


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

cv = CountVectorizer(ngram_range=(1, 2))

x_train_new = cv.fit_transform(x_train_clean)
x_test_new = cv.transform(x_test_clean)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=300)
model.fit(x_train_new, y_train)
predictions_training = model.predict(x_train_new)
print(classification_report(y_train, predictions_training))
predictions_testing = model.predict(x_test_new)
print(classification_report(y_test, predictions_testing))

def sentiment(text):
    trtext = TransformWords(text)
    trtext = cv.transform(trtext)
    textpred = model.predict(trtext)
    if textpred[0] == 1:
        print("Negative")
    else:
        print("Positive")

data1 = ["I ate Pizza for dinner and I hated it"]
data2 = ["I did not like doing my homework"]
data3 = ["I like going for shopping"]
data4 = ["I love playing with my friends"]
data5 = ["I'm nervous for my exam marks"]
sentiment(data1)
sentiment(data2)
sentiment(data3)
sentiment(data4)
sentiment(data5)
