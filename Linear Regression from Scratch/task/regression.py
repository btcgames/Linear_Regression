# write your code here
import math

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
                   'f2': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3],
                   'f3': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
                   'y': [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]})

X_train = df.loc[:, ['f1', 'f2', 'f3']]
y_train = df.y

regSci = LinearRegression(fit_intercept=True)
regSci.fit(X_train, y_train)
predictions_train = regSci.predict(X_train)
rmse_regSci = math.sqrt(mean_squared_error(y_train, predictions_train))
r2_regSci = r2_score(y_train, predictions_train)


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            self.add_ones(X)

            arr = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

            self.intercept = arr[0]
            self.coefficient = arr[1:]
        else:
            self.coefficient = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

    def predict(self, X):
        if self.fit_intercept:
            arr = self.coefficient.copy()
            arr = np.insert(arr, 0, self.intercept)
            return np.dot(X, arr)
        else:
            return np.dot(X, self.coefficient)

    def r2_score(self, y, yhat):
        y_mean = y.mean()
        return 1 - sum([(y_i - yhat_i)**2 for y_i, yhat_i in zip(y, yhat)]) / sum([(y_i - y_mean)**2 for y_i in y])

    def rmse(self, y, yhat):
        return math.sqrt((1/len(y)) * sum([(y_i - yhat_i)**2 for y_i, yhat_i in zip(y, yhat)]))

    def add_ones(self, X):
        ones = np.ones(len(X), dtype=int)
        X.insert(0, 'ones', ones)

regCustom = CustomLinearRegression(fit_intercept=True)
regCustom.fit(X_train, y_train)
y_pred = regCustom.predict(X_train)
r2 = regCustom.r2_score(y_train, y_pred)
rmse = regCustom.rmse(y_train, y_pred)

dic = dict(Intercept=(regSci.intercept_ - regCustom.intercept), Coefficient=regSci.coef_ - regCustom.coefficient,
           R2=r2_regSci - r2, RMSE=rmse_regSci - rmse)
print(dic)
