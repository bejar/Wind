"""
.. module:: RNNS2SArchitecture

RNNS2SArchitecture
******

:Description: RNNS2SArchitecture

    RNN with multople regression

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Flatten, Dropout
from sklearn.metrics import r2_score

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


class RNNS2SArchitecture(NNS2SArchitecture):
    """
    Recurrent architecture for sequence to sequence

    """
    modfile = None
    modname = 'RNNS2S'
    data_mode = ('3D', '2D')

    def generate_model(self):
        """
        Model for RNN for S2S multiple regression

        -------------
        json config:

        "arch": {
            "neurons":128,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0.3,
            "nlayers": 1,
            "activation": "tanh",
            "activation_r": "hard_sigmoid",
            "CuDNN": false,
            "bidirectional": false,
            "bimerge":"ave",
            "rnn": "GRU",
            "full": [64, 32],
            "activation_full": "sigmoid",
            "fulldrop": 0.05,
            "mode": "RNN_s2s"
        }

        :return:
        """
        neurons = self.config['arch']['neurons']
        drop = self.config['arch']['drop']
        nlayersE = self.config['arch']['nlayers']  # >= 1

        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        rec_reg = self.config['arch']['rec_reg']
        rec_regw = self.config['arch']['rec_regw']
        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']
        rnntype = self.config['arch']['rnn']

        full = self.config['arch']['full']
        fulldrop = self.config['arch']['fulldrop']
        activation_full = self.config['arch']['activation_full']

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
        self.model = Sequential()
        if nlayersE == 1:
            self.model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,return_sequences=True,
                               recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                               recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            self.model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                               recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                               return_sequences=True, recurrent_regularizer=rec_regularizer,
                               kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                self.model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                   activation=activation, recurrent_activation=activation_r, return_sequences=True,
                                   recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            self.model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,return_sequences=True,
                               recurrent_activation=activation_r, implementation=impl,
                               recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        self.model.add(Flatten())


        for nn in full:
            self.model.add(Dense(nn, activation=activation_full))
            self.model.add(Dropout(rate=fulldrop))

        self.model.add(Dense(odimensions))

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

