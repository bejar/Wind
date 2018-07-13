"""
.. module:: PersistenceArchitecture

PersistenceArchitecture
******

:Description: PersistenceArchitecture

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.Architecture import Architecture
from sklearn.metrics import r2_score

__author__ = 'bejar'

class PersistenceArchitecture(Architecture):

    def generate_model(self):
        """
        Generates the model

        :return:
        """
        pass

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        pass


    def summary(self):
        """
        Model summary
        :return:
        """
        print("Persitence")

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """
        r2persV = r2_score(val_y[1:], val_y[0:-1])
        r2persT = r2_score(test_y[:, 0], test_y[0:-1, 0])


    def log_result(self, result):
        """
        logs a result from the model

        :param result:
        :return:
        """
        for i, r2val, r2test in result:
            print('%s | DNM= %s, DS= %d, AH= %d, R2PV = %3.5f, R2PT = %3.5f' %
                  (self.config['arch']['mode'],
                   self.config['data']['datanames'][0],
                   self.config['data']['dataset'],
                   i,
                   r2val, r2test
                   ))


    def save(self, postfix):
        """
        Saves model to a file
        :return:
        """
        pass