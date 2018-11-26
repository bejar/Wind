"""
.. module:: RNNEncoderDecoderS2SArchitecture

RNNEncoderDecoderS2SArchitecture
******

:Description: RNNEncoderDecoderS2SAttentionArchitecture

    Encoder-decoder S2S with attention

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Bidirectional, Dense, TimeDistributed, Flatten, RepeatVector, Input
from sklearn.metrics import r2_score
from keras.layers import Activation, dot, concatenate

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


class RNNEncoderDecoderS2SArchitecture(NNS2SArchitecture):
    modfile = None
    modname = 'RNNEDS2SATT'
    data_mode = (False, '3D') # 's2s'

    def generate_model(self):
        """
        Model for RNN with Encoder Decoder for S2S with attention

        :return:
        """
        neurons = self.config['arch']['neurons']
        drop = self.config['arch']['drop']
        nlayersE = self.config['arch']['nlayersE']  # >= 1
        nlayersD = self.config['arch']['nlayersD']  # >= 1

        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        rec_reg = self.config['arch']['rec_reg']
        rec_regw = self.config['arch']['rec_regw']
        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']
        rnntype = self.config['arch']['rnn']
        CuDNN = self.config['arch']['CuDNN']
        neuronsD = self.config['arch']['neuronsD']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']
        impl = self.runconfig.impl

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



        RNN = LSTM if rnntype == 'LSTM' else GRU

        # Encoder RNN
        enc_input = Input(shape=(idimensions))
        encoder = RNN(neurons, implementation=impl,
                               recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r, unroll=True,
                               recurrent_regularizer=rec_regularizer,return_sequences=True, kernel_regularizer=k_regularizer)(enc_input)

        encoder_last = encoder[:,-1,:]

        # Decoder RNN
        dec_input = Input(shape=(odimensions))
        init_state = [encoder_last, encoder_last] if rnntype == 'LSTM' else [encoder_last]
        decoder = RNN(neuronsD, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      unroll=True, recurrent_regularizer=rec_regularizer,return_sequences=True,
                      kernel_regularizer=k_regularizer)(dec_input, initial_state=init_state)

        attention = dot([decoder, encoder], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)

        context = dot([attention, encoder], axes=[2,1])
        print('context', context)

        decoder_combined_context = concatenate([context, decoder])
        print('decoder_combined_context', decoder_combined_context)


        output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
        output = TimeDistributed(Dense(odimensions, activation="softmax"))(output)

        self.model = Model(inputs=[enc_input, dec_input], outputs=output)


    def evaluate(self, val_x, val_y, test_x, test_y):
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)
        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        ahead = self.config['data']['ahead']

        lresults = []
        for i in range(1, ahead + 1):
            lresults.append((i,
                             r2_score(val_y[:, i - 1], val_yp[:, i - 1]),
                             r2_score(test_y[:, i - 1], test_yp[:, i - 1])
                             ))
        return lresults

    def summary(self):
        self.model.summary()
        neurons = self.config['arch']['neurons']
        neuronsD = self.config['arch']['neuronsD']
        nlayersE = self.config['arch']['nlayersE']  # >= 1
        nlayersD = self.config['arch']['nlayersD']  # >= 1
        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        print('lag: ', self.config['data']['lag'], '/Neurons: ', neurons, neuronsD, '/Layers: ', nlayersE, nlayersD,
              '/Activation:', activation, activation_r)

    def log_result(self, result):
        for i, r2val, r2test in result:
            print(
                    '%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d %d, NN= %d %d, DR= %3.2f, AF= %s, RAF= %s, '
                    'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
                    (self.config['arch']['mode'],
                     self.config['data']['datanames'][0],
                     self.config['data']['dataset'],
                     len(self.config['data']['vars']),
                     self.config['data']['lag'],
                     i,
                     self.config['arch']['rnn'],
                     self.config['arch']['bimerge'] if self.config['arch']['bidirectional'] else 'no',
                     self.config['arch']['nlayersE'], self.config['arch']['nlayersD'],
                     self.config['arch']['neurons'], self.config['arch']['neuronsD'],
                     self.config['arch']['drop'],
                     self.config['arch']['activation'],
                     self.config['arch']['activation_r'],
                     self.config['training']['optimizer'],
                     r2val, r2test
                     ))


