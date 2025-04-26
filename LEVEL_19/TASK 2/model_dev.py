import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import mlflow
mlflow.sklearn.autolog()

from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import pandas as pd
from evaluation import MSE, R2 

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg

data = pd.DataFrame({
    'Suburb': ['Abbotsford'] * 28,
    'Address': ['85 Turner', '86 Turner', '87 Turner', '88 Turner', '89 Turner', '90 Turner', '91 Turner', '92 Turner', 
                '93 Turner', '94 Turner', '25 Bloom', '26 Bloom', '27 Bloom', '28 Bloom', '29 Bloom', '30 Bloom', 
                '31 Bloom', '32 Bloom', '33 Bloom', '34 Bloom', '5 Charles', '6 Charles', '7 Charles', '8 Charles', 
                '9 Charles', '10 Charles', '11 Charles', '12 Charles']
})

data['Target'] = np.random.rand(len(data))

X = pd.get_dummies(data[['Suburb', 'Address']]) 
y = data['Target']

model = LinearRegressionModel()
trained_model = model.train(X, y) 
y_pred = trained_model.predict(X) 

mse_eval = MSE()
r2_eval = R2()
mse = mse_eval.calculate_scores(y, y_pred) 
r2 = r2_eval.calculate_scores(y, y_pred)   
print(f"MSE: {mse}, R2: {r2}")
