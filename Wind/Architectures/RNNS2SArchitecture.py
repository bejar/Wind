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
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Dense, Flatten, Dropout, Bidirectional, Input
from sklearn.metrics import r2_score
from Wind.Train.Activations import generate_activation
from Wind.Train.Layers import generate_recurrent_layer

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
        bidir = self.config['arch']['bidirectional']
        bimerge = self.config['arch']['bimerge']

        if 'backwards' in self.config['arch']:
            backwards = self.config['arch']['backwards']
        else:
            backwards = False

        # GRU parameter for alternative implementation
        if 'after' in self.config['arch']:
            after = self.config['arch']['after']
        else:
            after = False

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

        # RNN = LSTM if rnntype == 'LSTM' else GRU

        input = Input(shape=(idimensions))

        if nlayersE == 1:
            model = generate_recurrent_layer(neurons, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidir, bimerge, rseq=True)(input)
            model = generate_activation(activation)(model)

        else:
            model = generate_recurrent_layer(neurons, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidir, bimerge, rseq=True)(input)
            model = generate_activation(activation)(model)


            for i in range(1, nlayersE - 1):
                model = generate_recurrent_layer(neurons, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                                 rnntype, after, bidir, bimerge, rseq=True)(model)
                model = generate_activation(activation)(model)


            model = generate_recurrent_layer(neurons, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidir, bimerge, rseq=True)(model)
            model = generate_activation(activation)(model)


        model = Flatten()(model)

        for nn in full:
            model = Dense(nn)(model)
            model = generate_activation(activation_full)(model)
            model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)


    # def evaluate(self, val_x, val_y, test_x, test_y):
    #     batch_size = self.config['training']['batch']
    #
    #     if self.runconfig.best:
    #         self.model = load_model(self.modfile)
    #     val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
    #     test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)
    #
    #     ahead = self.config['data']['ahead']
    #
    #     lresults = []
    #     for i in range(1, ahead + 1):
    #         lresults.append((i,
    #                          r2_score(val_y[:, i - 1], val_yp[:, i - 1]),
    #                          r2_score(test_y[:, i - 1], test_yp[:, i - 1])
    #                          ))
    #     return lresults
