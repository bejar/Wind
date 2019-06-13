"""
.. module:: Error

Error
*************

:Description: Error

    

:Authors: bejar
    

:Version: 

:Created on: 30/05/2019 14:19 

"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
__author__ = 'bejar'



def smape_error(y,yp):
    """
    Simetric Mean Absolute Percentage Error sMAPE
    :param y:
    :param yp:
    :return:
    """
    err = 2 * np.abs(y-yp)
    err /= (np.abs(y) + np.abs(yp))

    return (100.0/len(y)) * np.sum(err)

def mape_error(y,yp):
    """
    Simetric Mean Absolute Percentage Error sMAPE
    :param y:
    :param yp:
    :return:
    """
    err = np.abs(y-yp)
    err /= np.abs(y)

    return (100.0/len(y)) * np.sum(err)

class ErrorMeasure:
    """
    Compute the error measures from the data and predictions
    """
    error_names = None
    errors = ['R2', 'MSE', 'MAE']
    error_func = [r2_score, mean_squared_error, mean_absolute_error]

    def __init(self):
        """
        Initialization for the error object
        :return:
        """
        self.error_names = []
        for e in self.errors:
            self.error_names.append(e+'val')
            self.error_names.append(e+'test')


    def compute_errors(self, val_y, val_yp, test_y, test_yp):
        """

        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """

        lerr = []

        for f in self.error_func:
            lerr.append(f(val_y,val_yp))
            lerr.append(f(test_y, test_yp))

        return lerr


    def print_errors(self, arch, nres, result):
        """
        Print a list of errors, the structure of the list is elements
        that have the horizon as first element and then a list of errors (possible partial) in
        the order indicated by this class
        :param lerrors:
        :return:
        """
        nvals = len(result[0])
        count = [0]*nvals
        for c, (i, *err) in enumerate(result):
            print(f"{arch} | AH={i:2}", end=" ")
            for p, v in enumerate(err):
                count[p]+=v
                print(f"{ErrorMeasure.error_names[p]} = {v:3.5f}", end=" ")

            print()
            if (c+1) % nres == 0:
                print(f"**{arch} | AH=TT", end=" ")
                for p, v in enumerate(err):
                    print(f"{ErrorMeasure.error_names[p]} = {count[p]:3.5f}", end=" ")
                count = [0]*nvals
                print()

    def get_error(self, verrors, error):
        """
        Receives a matrix that contains for each row the horizon and the errors
        :param error:
        :return:
        """
        ind = self.errors.index(error)

        if verrors.shape[1] < (2*ind)+1:
            raise NameError(f'the error {error} has not been recorded')
        return verrors[:,(2*ind)+1], verrors[:,(2*ind)+2]