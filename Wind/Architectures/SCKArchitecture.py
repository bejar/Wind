"""
.. module:: SCKArchitecture

SCKArchitecture
*************

:Description: SCKArchitecture

 Metaclass for scikit learn classifiers using direct regression

:Authors: bejar
    

:Version: 

:Created on: 04/12/2018 7:46 

"""

from Wind.Architectures.Architecture import Architecture
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class SCKArchitecture(Architecture):
    """
    Class for all the scikit models using direct regression

    """
    ## data mode 2 dimensional input and only one output
    data_mode = ('2D', '0D') #'svm'
    modname = 'SCKDIRREG'

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        self.model.fit(train_x, train_y)

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """
        val_yp = self.model.predict(val_x)
        test_yp = self.model.predict(test_x)

        return ErrorMeasure().compute_errors(val_y, val_yp, test_y, test_yp)

        # r2val = r2_score(val_y, val_yp)
        # mseval = mean_squared_error(val_y, val_yp)
        # r2test = r2_score(test_y, test_yp)
        # msetest = mean_squared_error(test_y, test_yp)
        #
        # return [r2val, r2test, mseval, msetest]


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
        

