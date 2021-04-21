"""
.. module:: NaivePeriodicArchitecture

PersistenceArchitecture
******

:Description: NaivePeriodicArchitecture

    Class for Naive Periodic model

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.Architecture import Architecture
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class NaivePeriodicArchitecture(Architecture):
    """Class for naive periodic model

    """
    ## Data mode default for input, 1 dimensional output
    data_mode = ('2D', '2D')
    modname = 'NaivePeriodic'

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
        """Model summary

        prints all the fields stored in the configuration for the experiment

        :return:
        """
        print("--------- Architecture parameters -------")
        print(f"{self.modname}")
        for c in self.config['arch']:
            print(f"# {c} = {self.config['arch'][c]}")
        print("--------- Data parameters -------")
        for c in self.config['data']:
            print(f"# {c} = {self.config['data'][c]}")
        if 'training' in self.config:
            print("--------- Training parameters -------")
            for c in self.config['training']:
                print(f"# {c} = {self.config['training'][c]}")
            print("---------------------------------------")

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        Evaluates the training
        :param save_errors:
        :return:
        """

        if type(self.config['data']['ahead']) == list:
            ahead = self.config['data']['ahead'][1]
        else:
            ahead = self.config['data']['ahead']

        period = self.config['arch']['period']


        lresults = []
        for i in range(1, ahead + 1):
            lresults.append([i] + ErrorMeasure().compute_errors(val_y[period:, i - 1],
                                                                val_x[i-1:-period+i-1, -1],
                                                                test_y[period:, i - 1],
                                                                test_x[i-1:-period+i-1, -1]))
        return lresults
