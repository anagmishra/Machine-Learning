import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("adult_with_column_headings.csv")
 
data["income"] = data["income"].replace({"<=50K": 0, ">50K": 1})

x = data[["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]]
y = data["income"]

print(data.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print("Intercept: ", model.intercept_)

for feature,coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef:.4f}") #4f is upto 4 decimal points
    
print("Score calculation: ", r2_score(y_test, prediction))
print("Mean Squared Error: ", mean_squared_error(y_test, prediction)) # Squares the difference between the actual values and the prediction values (to avoid negatives which cannot be used on graph), then adds them, divides them by the number of values, and square roots them
results = pd.DataFrame({"Actual":y_test.values,
                        "Predicted": prediction})
print(results)