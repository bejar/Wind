"""
.. module:: TestModel

TestModel
*************

:Description: TestModel

    

:Authors: bejar
    

:Version: 

:Created on: 26/06/2018 13:18 

"""

from keras.models import Sequential, load_model
import os
import argparse
from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Config import wind_models_path, wind_data_path
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sec', help='Sites Section')
    parser.add_argument('--site', help='Initial Site')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--irange', type=int, help='Initial example')
    parser.add_argument('--erange', type=int, help='Final example')

    args = parser.parse_args()
    print('Config file: /%s/%s/%s' % (args.sec, args.site, args.config))
    config = load_config_file(wind_models_path + '/%s/%s/%s' % (args.sec, args.site, args.config), id=True,
                              abspath=True)

    if config['arch']['mode'] == 'regdir':
        datamode = False
        arch = 'RNNDir'
    elif config['arch']['mode'] == 'seq2seq':
        datamode = 's2s'
        arch = 'RNNS2S'
    elif config['arch']['mode'] == 'mlps2s':
        datamode = 'mlp'
        arch = 'MLPRegS2S'

    ahead = config['data']['ahead']
    lag = config['data']['lag']
    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=datamode,
                                                                      data_path=wind_data_path)
    batch_size = config['training']['batch']

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    lpreds = []
    for iter in range(niter):
        print('Testing: model%s-S%s-A%d-R%02d' % (arch, config['data']['datanames'][0], ahead, iter))
        model = load_model(
            wind_models_path + '/%s/%s/model%s-S%s-A%d-R%02d.h5' % (
            args.sec, args.site, arch, config['data']['datanames'][0], ahead, iter))
        lpreds.append(model.predict(val_x, batch_size=batch_size, verbose=0))

    for i in range(1, ahead + 1):
        vals = val_y[args.irange:args.erange, i - 1, 0]
        pred = np.stack([v[args.irange:args.erange, i - 1, 0].ravel() for v in lpreds])

        fig = plt.figure()

        axes = fig.add_subplot(1, 1, 1)
        plt.title('AHEAD=%d'%i)
        plt.plot(vals, 'r--')

        plt.plot(np.max(pred, axis=0), 'b')
        plt.plot(np.min(pred, axis=0), 'b')
        plt.plot(np.mean(pred, axis=0), 'g')
        plt.show()
