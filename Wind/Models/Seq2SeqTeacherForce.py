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

from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, TimeDistributed, Flatten, RepeatVector
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from time import time, strftime
import os
import numpy as np

__author__ = 'bejar'


def architectureS2STForce(ahead, idimensions, odimensions, neurons, neuronsD, drop, nlayersE, nlayersD, activation, activation_r, rnntype, CuDNN, impl=1):
    """

    :param ahead:
    :param idimensions:
    :param neurons:
    :param neuronsD:
    :param drop:
    :param nlayersE:
    :param nlayersD:
    :param activation:
    :param activation_r:
    :param rnntype:
    :param impl:
    :return:
    """
    RNN = LSTM #if rnntype == 'LSTM' else GRU
    encoder_inputs = Input(shape=(None, idimensions))
    encoder = RNN(neurons, return_state=True, implementation=impl,
                  recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, odimensions))
    decoder_lstm = RNN(neurons, return_sequences=True, return_state=True, implementation=impl,
                       recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(odimensions, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(neurons,))
    decoder_state_input_c = Input(shape=(neurons,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model


def train_seq2seq_tforce__architecture(config, impl, verbose, tboard, best, early):
    """
    Training process for RNN architecture with sequence to sequence regression of ahead time steps

    :return:
    """
    ahead = config['data']['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='s2s',
                                                                      data_path=wind_data_path)

    tfv = np.zeros((train_y.shape[0], 1, train_y.shape[2]))-3
    train_x_f = np.concatenate([tfv, train_y[:,:-1,:]], axis=1)
    tfv = np.zeros((val_y.shape[0], 1, val_y.shape[2]))-3
    val_x_f = np.concatenate([tfv, val_y[:,:-1,:]], axis=1)
    tfv = np.zeros((test_y.shape[0], 1, test_y.shape[2]))-3
    test_x_f = np.concatenate([tfv, test_y[:,:-1,:]], axis=1)

    print(train_x_f.shape)
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

    print(train_x.shape)
    model, inf_model_enc, inf_model_dec = architectureS2STForce(ahead=ahead,
                            idimensions= train_x.shape[-1],
                            odimensions=1,
                            neurons=neurons,
                            neuronsD=config['arch']['neuronsD'],
                            drop=drop, nlayersE=nlayersE,
                            nlayersD=nlayersD,
                            activation=activation, impl=impl,
                            activation_r=activation_r, rnntype=config['arch']['rnn'], CuDNN=config['arch']['CuDNN'])
    if verbose:
        model.summary()
        print('lag: ', config['data']['lag'], 'Neurons: ', neurons, 'Layers: ', nlayersE, nlayersD, activation,
              activation_r)
        print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
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

    optimizer = config['training']['optimizer']
    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'])
        else:
            optimizer = RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    batch_size = config['training']['batch']
    nepochs = config['training']['epochs']

    model.fit([train_x,train_x_f], train_y, batch_size=batch_size, epochs=nepochs, validation_data=([val_x, val_x_f], val_y),
              verbose=verbose, callbacks=cbacks)

    ############################################
    # Results

    if best:
        model = load_model(modfile)
    #
    # val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
    # test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)

    for i in range(100):
        output = list()
        state = inf_model_enc.predict(val_x[i].reshape(1, config['data']['lag'], 1))
        target_seq = np.array([0.0 for _ in range(1)]).reshape(1, 1, 1)
        for t in range(ahead):
            yhat, h, c = inf_model_dec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        print(np.array(output), val_y[i])


    lresults = []

    # for i in range(1, ahead + 1):
    #     lresults.append((i,
    #                      r2_score(val_y[:, i - 1, 0], val_yp[:, i - 1, 0]),
    #                      r2_score(val_y[i:, 0, 0], val_y[0:-i, 0, 0]),
    #                      r2_score(test_y[:, i - 1, 0], test_yp[:, i - 1, 0]),
    #                      r2_score(test_y[i:, 0, 0], test_y[0:-i, 0, 0])))
    #
    # for i, r2val, r2persV, r2test, r2persT in lresults:
    #     print('DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d %d, NN= %d %d, DR= %3.2f, AF= %s, RAF= %s, '
    #           'OPT= %s, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
    #           (config['data']['datanames'][0],
    #            config['data']['dataset'],
    #            len(config['data']['vars']),
    #            config['data']['lag'],
    #            i,
    #            config['arch']['rnn'],
    #            config['arch']['bimerge'] if config['arch']['bidirectional'] else 'no',
    #            config['arch']['nlayersE'],config['arch']['nlayersD'],
    #            config['arch']['neurons'],config['arch']['neuronsD'],
    #            config['arch']['drop'],
    #            config['arch']['activation'],
    #            config['arch']['activation_r'],
    #            config['training']['optimizer'],
    #            r2val, r2persV, r2test, r2persT
    #            ))


    try:
        os.remove(modfile)
    except OSError:
        pass

    return lresults


if __name__ == '__main__':

    from Wind.Util import load_config_file
    config = load_config_file("config.json")
    wind_data_path = '../../Data'
    lresults = train_seq2seq_tforce__architecture(config, 1, True, False, True, True)