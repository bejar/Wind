"""
.. module:: SVMRegression

SVMRegression
*************

:Description: SVMRegression

    

:Authors: bejar
    

:Version: 

:Created on: 25/04/2018 14:01 

"""

from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from Wind.Training import updateprocess
from time import time, strftime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


__author__ = 'bejar'

def train_svm_dirregression_architecture(config, verbose):
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

        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='svm',
                                                                          data_path=wind_data_path)

        # train_x = np.squeeze(train_x, axis=2)
        # val_x = np.squeeze(val_x, axis=2)
        # test_x = np.squeeze(test_x, axis=2)

        ############################################
        # Model

        kernel = config['arch']['kernel']
        C = config['arch']['C']
        epsilon = config['arch']['epsilon']
        degree = config['arch']['degree']
        coef0 = config['arch']['coef0']

        if verbose:

            print('lag: ', config['data']['lag'], '/kernel: ', kernel, '/C: ', C, '/epsilon:', epsilon, '/degree:', degree)
            print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape,
                  test_y.shape)
            print()

        ############################################
        # Training

        svmr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)
        svmr.fit(train_x, train_y)


        ############################################
        # Results

        val_yp = svmr.predict(val_x)

        r2val = r2_score(val_y, val_yp)
        r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])


        test_yp = svmr.predict(test_x)
        r2test = r2_score(test_y, test_yp)
        r2persT = r2_score(test_y[ahead:], test_y[0:-ahead])

        lresults.append((ahead, r2val, r2persV, r2test, r2persT))
        print('%s |  AH=%d, KRNL= %s, C= %3.5f, EPS= %3.5f, DEG=%d, COEF0= %d, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
              (config['arch']['mode'], ahead,
        config['arch']['kernel'],
        config['arch']['C'],
        config['arch']['epsilon'],
        config['arch']['degree'],
        config['arch']['coef0'],
               r2val, r2persV, r2test, r2persT
               ))
        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is None:
            updateprocess(config, ahead)


    return lresults

if __name__ == '__main__':

    from Wind.Util import load_config_file
    config = load_config_file("configsvmdir.json")
    wind_data_path = '../../Data'
    lresults = train_svm_dirregression_architecture(config, False)