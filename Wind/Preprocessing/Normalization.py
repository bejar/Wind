"""
.. module:: Normalization

Normalization
*************

:Description: Normalization

    Classes for data normalization

:Authors: bejar
    

:Version: 

:Created on: 14/12/2018 7:22 

"""

import numpy as np

__author__ = 'bejar'


class tanh_normalization:
    """
    Tanh normalization
    """
    mu = 0.0
    sigma = 0.0

    def __init__(self):
        """
        normalization
        """
        pass

    def fit(self,X):
        """
        Just computes the parameters for the normalization
        :param X:
        :return:
        """
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)

    def transform(self, X):
        """
        Computes the tanh transformation
        :param X:
        :return:
        """
        return 0.5 * (np.tanh(0.01 * (X - self.mu) /self.sigma) + 1)

    def fit_transform(self, X):
        """
        Fit and transform at the same time
        :param X:
        :return:
        """
        self.fit(X)
        return self.transform(X)


class sigmoid_normalization:
    """
    sigmoid normalization
    """
    mu = 0.0
    sigma = 0.0

    def __init__(self):
        """
        normalization
        """
        pass

    def fit(self,X):
        """
        nothing to do for fitting
        :param X:
        :return:
        """
        pass

    def transform(self,X):
        """
        Computes the tanh transformation
        :param X:
        :return:
        """
        tmp = np.exp(-X)
        print(tmp.shape)
        return 1.0/(1.0 - tmp)

    def fit_transform(self, X):
        """
        Fit and transform at the same time
        :param X:
        :return:
        """
        return self.transform(X)