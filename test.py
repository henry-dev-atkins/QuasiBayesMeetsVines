import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from model.QBVM import QBV
import pickle

np.random.seed(42)

if __name__ == '__main__':
    df = load_diabetes(return_X_y = False, as_frame = True).frame
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    #qbv = QBV(p0_class='Cauchy')
    #qbv.fit(df)
    #qbv.save_model('henryTest')

    qbv = QBV(p0_class='Cauchy')
    qbv = qbv.load_model('henryTest')

    y_pred = qbv.predict(X.iloc[:10], 0.000000001, 0.999999999999)
    print(y_pred)
    y = y[:10]
    print(y.to_numpy().shape)
    mse = mean_squared_error(y.to_numpy(), y_pred.numpy())

    qbv.save_model('henry_test')

    print('MSE:', mse)
    print('Finished seed', 42)