"""
.. module:: Seq2SeqRegression

Seq2SeqRegression
*************

:Description: Seq2SeqRegression

  Sequence 2 sequence RNN regression

:Authors: bejar

:Version: 

:Created on: 06/04/2018 14:27 

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Flatten, RepeatVector
try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True
try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2



import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from time import time, strftime
import os

__author__ = 'bejar'


def architectureS2S(idimensions, odimensions, neurons, neuronsD, drop, nlayersE, nlayersD, activation, activation_r,
                    rnntype, impl=1, CuDNN=False, bidirectional=False, rec_reg='l1', rec_regw=0.1, k_reg='l1',
                    k_regw=0.1):
    """

    Sequence to sequence architecture

    :param neurons:
    :param drop:
    :param nlayers:
    :param activation:
    :param activation_r:
    :param rnntype:
    :param CuDNN:
    :param bidirectional:
    :param rec_reg:
    :param rec_regw:
    :param k_reg:
    :param k_regw:
    :return:
    """

    if rec_reg == 'l1':
        rec_regularizer = l1(rec_regw)
    elif rec_reg == 'l2':
        rec_regularizer = l2(rec_regw)
    else:
        rec_regularizer = None

    if k_reg == 'l1':
        k_regularizer = l1(k_regw)
    elif rec_reg == 'l2':
        k_regularizer = l2(k_regw)
    else:
        k_regularizer = None

    if CuDNN:
        RNN = CuDNNLSTM if rnntype == 'LSTM' else CuDNNGRU
        model = Sequential()
        if nlayersE == 1:

            model.add(
                RNN(neurons, input_shape=(idimensions),
                    recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(idimensions), return_sequences=True,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                              kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(odimensions))

        for i in range(nlayersD):
            model.add(RNN(neuronsD, return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))
    else:
        RNN = LSTM if rnntype == 'LSTM' else GRU
        model = Sequential()
        if nlayersE == 1:
            model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                              activation=activation, recurrent_activation=activation_r, return_sequences=True,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                          recurrent_activation=activation_r, implementation=impl,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(odimensions))

        for i in range(nlayersD):
            model.add(RNN(neuronsD, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))

    return model


def train_seq2seq_architecture(config, impl, verbose, tboard, best, early, multi=1, save=False):
    """
    Training process for RNN architecture with sequence to sequence regression of ahead time steps

    :return:
    """
    ahead = config['data']['ahead']

    if not type(ahead) == list:
        ahead = [1, ahead]

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='s2s',
                                                                      data_path=wind_data_path)

    ############################################
    # Model

    neurons = config['arch']['neurons']
    drop = config['arch']['drop']
    nlayersE = config['arch']['nlayersE']  # >= 1
    nlayersD = config['arch']['nlayersD']  # >= 1

    activation = config['arch']['activation']
    activation_r = config['arch']['activation_r']
    rec_reg = config['arch']['rec_reg']
    rec_regw = config['arch']['rec_regw']
    k_reg = config['arch']['k_reg']
    k_regw = config['arch']['k_regw']
    batch_size = config['training']['batch']
    nepochs = config['training']['epochs']
    optimizer = config['training']['optimizer']

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] +1
    else:
        odimensions = ahead

    lresults = []
    for iter in range(niter):
        model = architectureS2S(idimensions=train_x.shape[1:], odimensions=odimensions, neurons=neurons,
                                neuronsD=config['arch']['neuronsD'], drop=drop, nlayersE=nlayersE, nlayersD=nlayersD,
                                activation=activation, activation_r=activation_r, rnntype=config['arch']['rnn'],
                                impl=impl, CuDNN=config['arch']['CuDNN'], rec_reg=rec_reg, rec_regw=rec_regw,
                                k_reg=k_reg, k_regw=k_regw)
        if verbose:
            model.summary()
            print('lag: ', config['data']['lag'], 'Neurons: ', neurons, 'Layers: ', nlayersE, nlayersD, activation,
                  activation_r)
            print(
            'Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
            print()

        ############################################
        # Training
        cbacks = []
        if tboard:
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            cbacks.append(tensorboard)

        if best:
            modfile = './model%d.h5' % int(time())
            mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
            cbacks.append(mcheck)

        if early:
            early = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
            cbacks.append(early)


        if optimizer == 'rmsprop':
            if 'lrate' in config['training']:
                optimizer = RMSprop(lr=config['training']['lrate'])
            else:
                optimizer = RMSprop(lr=0.001)



        if multi == 1:
            model.compile(loss='mean_squared_error', optimizer=optimizer)
        else:
            pmodel = multi_gpu_model(model, gpus=multi)
            pmodel.compile(loss='mean_squared_error', optimizer=optimizer)



        if multi == 1:
            model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=verbose, callbacks=cbacks)
        else:
            pmodel.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=verbose, callbacks=cbacks)

        ############################################
        # Results

        if best:
            model = load_model(modfile)

        val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)


        for i in range(ahead[0], ahead[1] + 1):
            lresults.append((i,
                             r2_score(val_y[:, i  - ahead[0], 0], val_yp[:, i - ahead[0], 0]),
                             # r2_score(val_y[i:, 0, 0], val_y[0:-i, 0, 0]),
                             r2_score(test_y[:, i - ahead[0] , 0], test_yp[:, i - ahead[0], 0]),
                             # r2_score(test_y[i:, 0, 0], test_y[0:-i, 0, 0])
                             )
                            )

        print(strftime('%Y-%m-%d %H:%M:%S'))


        if not save and best:
            try:
                os.remove(modfile)
            except OSError:
                pass
        elif best:
            os.rename(modfile, 'modelRNNS2S-S%s-A%d-%d-R%02d.h5'%(config['data']['datanames'][0], ahead[0], ahead[1], iter))

    for i, r2val, r2test in lresults:
        print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d %d, NN= %d %d, DR= %3.2f, AF= %s, RAF= %s, '
              'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               len(config['data']['vars']),
               config['data']['lag'],
               i,
               config['arch']['rnn'],
               config['arch']['bimerge'] if config['arch']['bidirectional'] else 'no',
               config['arch']['nlayersE'],config['arch']['nlayersD'],
               config['arch']['neurons'],config['arch']['neuronsD'],
               config['arch']['drop'],
               config['arch']['activation'],
               config['arch']['activation_r'],
               config['training']['optimizer'],
               r2val, r2test
               ))

    return lresults
