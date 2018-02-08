"""
.. module:: WindPredictionTS

WindPredictionTS
*************

:Description: WindPredictionTS

    

:Authors: bejar
    

:Version: 

:Created on: 26/01/2018 14:33 

"""
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.ar_model import AR
import json
import argparse
from Wind.Util import load_config_file
from sklearn.metrics import mean_squared_error, r2_score

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config1', help='Experiment configuration')
    args = parser.parse_args()

    config = load_config_file(args.config)

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'pressure'}

    wind = np.load('../Data/Wind60.npz')

    datasize = config['datasize']
    testsize = config['testsize']

    wind1 = wind['90-45142']
    print(wind1.shape)
    train = wind1[datasize:, 0].reshape(-1, 1)

    halftest = testsize/2
    test = wind1[datasize:datasize+halftest, 0].reshape(-1, 1)
    print(test.shape)

    tlength = 100
    plength = 8
    # npred = test.shape[0] - tlength - plength
    npred = 100
    predictions = np.zeros((npred, plength))


    for i in range(npred):
        arima = ARIMA(test[i:i+tlength], order=(12, 0, 0))
        # arima = ARMA(test[i:i+tlength], order=(12, 0))
        # arima = AR(np.array(test[i:i+tlength]))
        res = arima.fit(disp=0)
        fore = res.forecast(steps=plength)[0]
        predictions[i] = fore

    fig = plt.figure()
    plt.plot(test[tlength+7:tlength+npred+7], c='r')
    plt.plot(predictions[:, 7])
    plt.show()

    for i in range(plength):
        print('R2 val= %d' % i, r2_score(test[tlength+i:tlength+npred+i], predictions[:, i]))



