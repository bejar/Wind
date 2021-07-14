"""
.. module:: Error

Error
*************

:Description: Error

    

:Authors: bejar
    

:Version: 

:Created on: 30/05/2019 14:19 

"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

__author__ = 'bejar'


def smape_error(y, yp):
    """
    Symmetric Mean Absolute Percentage Error sMAPE
    :param y:
    :param yp:
    :return:
    """
    err = 2 * np.abs(y - yp)
    err /= (np.abs(y) + np.abs(yp))

    return (100.0 / len(y)) * np.sum(err)


def mape_error(y, yp):
    """
    Mean Absolute Percentage Error MAPE
    :param y:
    :param yp:
    :return:
    """
    err = np.abs(y - yp)
    err /= np.abs(y)

    return (100.0 / len(y)) * np.sum(err)


def mase_error(y, yp):
    """
    Mean Absolute Scaled Error
    Using a period of 24 hours
    :param y:
    :param yp:
    :return:
    """
    mae = mean_absolute_error(y, yp)
    q = np.sum(np.abs(y[24:] - y[0:-24])) / (len(y) - 24)
    return mae / q


def nrmse_error(y, yp):
    """
    Normalize Root Mean Square Error
    :param y:
    :param yp:
    :return:
    """

    ymax = np.max(y)
    ymin = np.min(y)
    if ymin < 0:
        ymax = ymax - ymin

    # return (100.0/ymax) * np.sqrt(np.sum((y-yp)**2)/len(y))
    return (100.0 / ymax) * rmse_error(y, yp)


def nmae_error(y, yp):
    """
    Normalize Mean Square Error
    :param y:
    :param yp:
    :return:
    """

    ymax = np.max(y)
    ymin = np.min(y)
    if ymin < 0:
        ymax = ymax - ymin

    # return (100.0/ymax) * np.sum(np.abs(y-yp)/len(y))
    return (100.0 / ymax) * mean_absolute_error(y, yp)


def rmse_error(y, yp):
    """
    Root Mean Square Error
    :param y:
    :param yp:
    :return:
    """
    return np.sqrt(mean_squared_error(y, yp))


class ErrorMeasure:
    """
    Compute the error measures from the data and predictions
    """
    error_names = []
    errors = ['R2', 'MSE', 'MAE', 'RMSE', 'nRMSE', 'nMAE', 'sMAPE', 'MASE']
    error_dict = {'R2': r2_score,
                  'MSE': mean_squared_error,
                  'RMSE': rmse_error,
                  'nRMSE': nrmse_error,
                  'MAE': mean_absolute_error,
                  'nMAE': nmae_error,
                  'sMAPE': smape_error,
                  'MAPE': mape_error,
                  'MASE': mase_error}
    error_func = []

    def __init__(self):
        """
        Initialization for the error object
        :return:
        """
        self.error_names = []
        self.error_func = []
        for e in self.errors:
            self.error_names += [f'{e}val', f'{e}test']
            self.error_func += [self.error_dict[e]]

    def compute_errors(self, val_y, val_yp, test_y, test_yp, scaler=None):
        """

        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """

        lerr = []

        if scaler is not None:
            for f in self.error_func:
                lerr.append(
                    np.float64(f(scaler.inverse_transform(val_y.reshape(-1, 1))), np.float64(scaler.inverse_transform(val_yp.reshape(-1, 1)))))
                lerr.append(np.float64(f(scaler.inverse_transform(test_y.reshape(-1, 1))),
                              np.float64(scaler.inverse_transform(test_yp.reshape(-1, 1)))))
        else:
            for f in self.error_func:
                lerr.append(np.float64(f(val_y, val_yp)))
                lerr.append(np.float64(f(test_y, test_yp)))

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
        count = [0] * nvals
        for c, (i, *err) in enumerate(result):
            print(f"{arch} | AH={i:2}", end=" ")
            for p, v in enumerate(err):
                count[p] += v
                print(f"{self.error_names[p]} = {v:3.5f}", end=" ")

            print()
            if (c + 1) % nres == 0:
                print(f"**{arch} | AH=TT", end=" ")
                for p, v in enumerate(err):
                    print(f"{self.error_names[p]} = {count[p]:3.5f}", end=" ")
                count = [0] * nvals
                print()

    def get_error(self, verrors, error):
        """
        Receives a matrix that contains for each row the horizon and the errors
        :param error:
        :return:
        """
        ind = self.errors.index(error)

        if verrors.shape[1] < (2 * ind) + 1:
            raise NameError(f'the error {error} has not been recorded')
        return verrors[:, (2 * ind) + 1], verrors[:, (2 * ind) + 2]
