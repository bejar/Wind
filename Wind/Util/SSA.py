"""
.. module:: SSA

SSA
*************

:Description: SSA

    

:Authors: bejar
    

:Version: 

:Created on: 30/06/2017 8:51 

"""
from scipy.linalg import hankel, svd
import numpy as np

__author__ = 'bejar'


class SSA:
    """
    Class for computing Singular Spectrum Analysis
    """
    n_components = 4
    Xi = None
    explained = None
    s = None
    length = 0
    components = None

    def __init__(self, n_components):
        """
        Creates the initial structure
        :param n_components:
        """
        self.n_components = n_components

    def fit(self, X):
        """
        Computes the embedding X is the time series

        :param X:
        :return:
        """
        if not (2 <= self.n_components < X.shape[0]):
            raise NameError('Series too short')

        self.length = X.shape[0]
        X = np.concatenate((X, np.zeros(self.n_components - 1)))

        mSSA = hankel(X[0:self.n_components], X[self.n_components - 1:])

        mSSA2 = np.dot(mSSA, mSSA.T)

        U, s, _ = svd(mSSA2)

        self.s = s

        self.explained = np.zeros(s.shape)
        for i in range(self.explained.shape[0]):
            self.explained[i] = s[i] / np.sum(s)

        lVi = []
        for i in range(self.n_components):
            lVi.append(np.dot(mSSA.T, U[i]) / np.sqrt(s[i]))

        lXi = []
        for i in range(self.n_components):
            lXi.append(np.sqrt(s[i]) * np.outer(U[i], lVi[i].T))

        self.Xi = lXi

    # TODO: error checking
    def decomposition(self, groups):
        """
        groups is a list of list that defines
        :param groups:
        :return:
        """

        ldecomp = []
        for g in groups:
            if type(g) == list:
                Xg = np.zeros(self.Xi[0].shape)
                for v in g:
                    Xg += self.Xi[v]
            elif type(g) == int:
                Xg = self.Xi[g]
            else:
                raise NameError('groups list incorrect')
            tmp = np.zeros(self.length)
            for k in range(self.length):
                if k < Xg.shape[0] - 1:
                    val = 0.0
                    for m in range(k + 1):
                        val += Xg[m, k - m]
                    tmp[k] = val / (k + 1)
                elif k < Xg.shape[1]:
                    val = 0.0
                    for m in range(Xg.shape[0]):
                        val += Xg[m, k - m]
                    tmp[k] = val / Xg.shape[0]
                else:
                    val = 0.0
                    for m in range(k - Xg.shape[1] + 1, self.length - Xg.shape[1] + 1):
                        val += Xg[m, k - m]
                    tmp[k] = val / (self.length - Xg.shape[1])
            ldecomp.append(tmp)
        self.components = np.array(ldecomp)
        return ldecomp

    def reconstruct(self, n):
        """
        Reconstructs the signal using n components
        :param n:
        :return:
        """
        return np.sum(self.components[0:n, ], axis=0)
