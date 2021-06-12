"""
.. module:: RNNEncoderDecoderS2DepSArchitecture

RNNEncoderDecoderS2SArchitecture
******

:Description: RNNEncoderDecoderS2SDepArchitecture

    RNN Encoder Decoder separating the encoder for dependent variables and auxiliary variables

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Architectures.Util import recurrent_encoder_functional, recurrent_decoder_functional
from keras.models import load_model, Model
from keras.layers import LSTM, GRU, Dense, TimeDistributed, RepeatVector, Input, concatenate
from sklearn.metrics import r2_score

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.regularizers import l1, l2

__author__ = 'bejar'


class RNNEncoderDecoderS2SDepArchitecture(NNS2SArchitecture):
    """
    Recurrent encoder decoder that separates the dependent variable from the rest

    """
    modfile = None
    modname = 'RNNEDS2SDep'
    data_mode = (False, '3D')

    def generate_model(self):
        """
        Model for RNN with Encoder Decoder for S2S separating the dependent variable from the auxiliary variables

        -------------
        json config:

        "arch": {
            "neuronsE":128,
            "neuronsD":64,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0.3,
            "nlayersE": 1,
            "nlayersD": 1,
            "activation": "relu",
            "activation_r": "hard_sigmoid",
            #"CuDNN": false,
            #"bidirectional": false,
            #"bimerge":"ave",
            "rnn": "GRU",
            "mode": "RNN_ED_s2s_dep"
        }
        ------------
        :return:
        """
        neuronsE = self.config['arch']['neuronsE']
        neuronsD = self.config['arch']['neuronsD']
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

        # Dependent variable input
        enc_Dep_input = Input(shape=(idimensions[0]))
        rec_Dep_input = recurrent_encoder_functional(RNN, nlayersE,
                                                    neuronsE, impl, drop,
                                                    activation, activation_r,
                                                    rec_regularizer, k_regularizer, enc_Dep_input)

        # Auxiliary variables input
        enc_Aux_input = Input(shape=(idimensions[1]))
        rec_Aux_input = recurrent_encoder_functional(RNN, nlayersE,
                                                    neuronsE, impl, drop,
                                                    activation, activation_r,
                                                    rec_regularizer, k_regularizer, enc_Aux_input)


        enc_input = concatenate([rec_Dep_input, rec_Aux_input])

        output = RepeatVector(odimensions)(enc_input)

        output = recurrent_decoder_functional(RNN, nlayersD,
                                                    neuronsD, impl, drop,
                                                    activation, activation_r,
                                                    rec_regularizer, k_regularizer, output)

        output = TimeDistributed(Dense(1))(output)

        self.model = Model(inputs=[enc_Dep_input, enc_Aux_input], outputs=output)

