"""
.. module:: NNS2SArchitecture

NNS2SArchitecture
*************

:Description: NNS2SArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 19/10/2018 10:32 

"""

from sklearn.metrics import r2_score, mean_squared_error
from keras.models import load_model

from Wind.Architectures.NNArchitecture import NNArchitecture
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class NNS2SArchitecture(NNArchitecture):
    """
    Class for all the neural networks models based on sequence to sequence

    """
    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None):
        """
        Evaluates the trained model with validation and test

        Overrides parent function

        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)
        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        # Maintained to be compatible with old configuration files
        if type(self.config['data']['ahead'])==list:
            iahead = self.config['data']['ahead'][0]
            ahead = (self.config['data']['ahead'][1] - self.config['data']['ahead'][0]) + 1
        else:
            iahead = 1
            ahead = self.config['data']['ahead']

        if 'aggregate' in self.config['data'] and 'y' in self.config['data']['aggregate']:
            step = self.config['data']['aggregate']['y']['step']
            ahead //= step

        lresults = []

        for i, p in zip(range(1, ahead + 1), range(iahead, self.config['data']['ahead'][1]+1)):
            lresults.append([p]  + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                                val_yp[:, i - 1],
                                                                test_y[:, i - 1],
                                                                test_yp[:, i - 1],
                                                                scaler=scaler))
        return lresults
