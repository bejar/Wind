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
    parser.add_argument('--byinstance', action='store_true', default=False, help='Visualize by instance')
    parser.add_argument('--meandev', action='store_true', default=False, help='Visualize mean and deviation')

    args = parser.parse_args()
    print(f"Config file: /{args.sec}/{args.site}/{args.config}")
    config = load_config_file(wind_models_path + f"/{args.sec}/{args.site}/{args.config}", id=True,
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

        if type(ahead) == list:
            fmodel = f"model{arch}-S{config['data']['datanames'][0]}-A{ahead[0]}-{ahead[1]}-R{iter:02d}"
        else:
            fmodel = f"model%{arch}-S{config['data']['datanames'][0]}-A{ahead}-R{iter:02d}"


        print(f"Testing: {fmodel}")
        model = load_model(
            wind_models_path + f"/{args.sec}/{args.site}/{fmodel}.h5")
        lpreds.append(model.predict(val_x, batch_size=batch_size, verbose=0))


    if not args.byinstance:
        for i in range(1, ahead + 1):
            if config['arch']['mode'] == 'seq2seq':
                vals = val_y[args.irange:args.erange, i - 1, 0]
                pred = np.stack([v[args.irange:args.erange, i - 1, 0].ravel() for v in lpreds])
            elif config['arch']['mode'] == 'mlps2s':
                vals = val_y[args.irange:args.erange, i - 1]
                pred = np.stack([v[args.irange:args.erange, i - 1].ravel() for v in lpreds])

            fig = plt.figure()

            axes = fig.add_subplot(1, 1, 1)
            plt.title(f"{fmodel} AHEAD={i}")
            plt.plot(vals, 'r--')

            plt.plot(np.max(pred, axis=0), 'b')
            plt.plot(np.min(pred, axis=0), 'b')
            plt.plot(np.mean(pred, axis=0), 'g')
            plt.show()
    else:
        for i in range(args.irange,args.erange):
            if config['arch']['mode'] == 'seq2seq':
                vals = val_y[i, :, 0]
                vals_x = val_x[i, :, 0]
                pred = np.stack([v[i, :, 0].ravel() for v in lpreds])
            elif config['arch']['mode'] == 'mlps2s':
                vals = val_y[i, :]
                vals_x = val_x[i, ::len(config['data']['vars'])]
                pred = np.stack([v[i, :].ravel() for v in lpreds])

            fig = plt.figure()

            axes = fig.add_subplot(1, 1, 1)
            plt.title(f"{fmodel} AHEAD={i}'")
            plt.plot(range(vals_x.shape[0]), vals_x, 'r')
            plt.plot(range(vals_x.shape[0],vals_x.shape[0]+vals.shape[0]), vals, 'r--')

            if args.meandev:
                plt.plot(range(vals_x.shape[0],vals_x.shape[0]+vals.shape[0]),np.max(pred, axis=0), 'b')
                plt.plot(range(vals_x.shape[0],vals_x.shape[0]+vals.shape[0]),np.min(pred, axis=0), 'b')
                plt.plot(range(vals_x.shape[0],vals_x.shape[0]+vals.shape[0]),np.mean(pred, axis=0), 'g')
            else:
                for j in range(pred.shape[0]):
                    plt.plot(range(vals_x.shape[0],vals_x.shape[0]+vals.shape[0]),pred[j], 'b')
            plt.show()
