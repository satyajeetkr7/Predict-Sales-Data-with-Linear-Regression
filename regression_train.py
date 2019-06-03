import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DataSet Load
data = pd.read_csv("F:\\E\\Mtech\\Datasets\\rossmann\\train.csv")
print(data.head())
print(data.describe())
print(data.dtypes)
print(data.StateHoliday.unique())
data.StateHoliday = data.StateHoliday.astype(str)

def count_unique(column):
    return len(column.unique())
print(data.apply(count_unique, axis=0).astype(np.int32))

print(data.isnull().any())

store_data = data[data.Store==150].sort_values('Date')
plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_data.Sales.values[:365])
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(x=store_data[data.Open==1].Promo, y=store_data[data.Open==1].Sales, alpha=0.1)
plt.show()

transformed_data = data.drop(['Store', 'Date', 'Customers'], axis=1)
transformed_data = pd.get_dummies(transformed_data, columns=['DayOfWeek', 'StateHoliday'])

X = transformed_data.drop(['Sales'], axis=1).values
y = transformed_data.Sales.values
print("The training dataset has {} examples and {} features.".format(X.shape[0], X.shape[1]))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
#kfolds = KFold(X.shape[0], n_folds=4, shuffle=True, random_state=42)
kfolds = KFold(n_splits=4, random_state=42, shuffle=True)
scores = cross_val_score(lr, X, y, cv=kfolds)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean()*100, scores.std()*100))
X_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values
y_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).Sales.values
lr.fit(X_store, y_store)
y_store_predict = lr.predict(pd.get_dummies(store_data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values)

plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_data.Sales.values[:365], label="ground truth")
plt.plot(y_store_predict[:365], c='r', label="prediction")
plt.legend()
plt.show()