"""
.. module:: Persistence

Persistence
*************

:Description: Persistence

    Computes the persistence predictor

:Authors: bejar
    

:Version: 

:Created on: 06/06/2018 14:58 

"""

from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from Wind.Training import updateprocess
from time import time, strftime
import os



__author__ = 'bejar'


def train_persistence(config, verbose, remote):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    for ahead in range(iahead, sahead + 1):

        if verbose:
            print('-----------------------------------------------------------------------------')
            print('Steps Ahead = %d ' % ahead)

        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
                                                                          data_path=wind_data_path,remote=remote)

        r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])

        r2persT = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])

        lresults.append((ahead, r2persV, r2persT))

        print('%s | DNM= %s, DS= %d, AH= %d, R2PV = %3.5f, R2PT = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               ahead,
               r2persV, r2persT
               ))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        # if config is not None:
        #     updateprocess(config, ahead)


        del train_x, train_y, test_x, test_y, val_x, val_y


    return lresults


if __name__ == '__main__':

    from Wind.Util import load_config_file
    config = load_config_file("configpersistence.json")
    wind_data_path = '../../Data'
    lresults = train_persistence(config, False)