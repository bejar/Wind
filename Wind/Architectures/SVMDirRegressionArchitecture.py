"""
.. module:: SVMDirRegressionArchitecture

SVMDirRegressionArchitecture
*************

:Description: SVMDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:33 

"""

from Wind.Architectures.Architecture import Architecture
from sklearn.svm import SVR
from sklearn.metrics import r2_score

__author__ = 'bejar'

class SVMDirRegressionArchitecture(Architecture):

    data_mode = 'svm'

    def generate_model(self):
        """
        Generates the model

        :return:
        """
        kernel = self.config['arch']['kernel']
        C = self.config['arch']['C']
        epsilon = self.config['arch']['epsilon']
        degree = self.config['arch']['degree']
        coef0 = self.config['arch']['coef0']

        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)


    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        self.model.fit(train_x, train_y)


    def summary(self):
        """
        Model summary
        :return:
        """
        kernel = self.config['arch']['kernel']
        C = self.config['arch']['C']
        epsilon = self.config['arch']['epsilon']
        degree = self.config['arch']['degree']
        coef0 = self.config['arch']['coef0']

        print('lag: ', self.config['data']['lag'], '/kernel: ', kernel, '/C: ', C, '/epsilon:', epsilon, '/degree:', degree)


    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """
        val_yp = self.model.predict(val_x)

        r2val = r2_score(val_y, val_yp)

        test_yp = self.model.predict(test_x)
        r2test = r2_score(test_y, test_yp)

        return r2val, r2test


    def log_result(self, result):
        """
        logs a result from the model

        :param result:
        :return:
        """
        for r in result:
            print('%s |  AH=%d, KRNL= %s, C= %3.5f, EPS= %3.5f, DEG=%d, COEF0= %d, R2V = %3.5f, R2T = %3.5f' %
                    (self.config['arch']['mode'], r[0],
                     self.config['arch']['kernel'],
                     self.config['arch']['C'],
                     self.config['arch']['epsilon'],
                     self.config['arch']['degree'],
                     self.config['arch']['coef0'],
                     r[1], r[2]
                     ))

    def save(self, postfix):
        """
        Saves model to a file
        :return:
        """
        pass