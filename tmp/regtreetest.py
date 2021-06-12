"""
.. module:: regtreetest

regtreetest
******

:Description: regtreetest

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  19/04/2019
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np

__author__ = 'bejar'

if __name__ == '__main__':

    X, y = make_regression(n_samples=1500, n_features=5, n_informative=5)
    Xtrain, Xtest = X[:1400,:], X[1400:,:]
    ytrain, ytest = y[:1400], y[1400:]

    print(X.shape, y.shape)

    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(Xtrain, ytrain)

    lpred=[]
    for tree in rfr.estimators_:
        lpred.extend(tree.predict(Xtest[0,:].reshape(1, -1)))

    print(np.mean(lpred), np.std(lpred), ytest[0], rfr.predict(Xtest[0,:].reshape(1, -1)))





