"""
.. module:: RNNEncoderDecoderS2SArchitecture

RNNEncoderDecoderS2SArchitecture
******

:Description: RNNEncoderDecoderS2SArchitecture

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Dense, TimeDistributed, RepeatVector, Dropout, Input, Concatenate
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


__author__ = 'bejar'


class RNNEncoderDecoderS2SArchitecture(NNS2SArchitecture):
    """
    Recurrent decoder encoder sequence to sequence architecture

    """
    modfile = None
    modname = 'RNNEDS2S'
    data_mode = ('3D', '3D') #'s2s'

    def generate_model(self):
        """
        Model for RNN with Encoder Decoder for S2S
        -------------
        json config:

        "arch": {
            "neuronsE":32,
            "neuronsD":16,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0.3,
            "nlayers": 2,
            "nlayersE": 1,
            "nlayersD": 1,
            "activation": "relu",
            "activation_r": "hard_sigmoid",
            "bidirectiolal":[false, false] <- bidirectional [encoder, decoder]
            "CuDNN": false,
            "rnn": "GRU",
            "mode": "RNN_ED_s2s"
        }

        :return:
        """
        neuronsE = self.config['arch']['neuronsE']
        neuronsD = self.config['arch']['neuronsD']
        nlayersE = self.config['arch']['nlayersE']  # >= 1
        nlayersD = self.config['arch']['nlayersD']  # >= 1
        drop = self.config['arch']['drop']

        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        rec_reg = self.config['arch']['rec_reg']
        rec_regw = self.config['arch']['rec_regw']
        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']
        rnntype = self.config['arch']['rnn']
        CuDNN = self.config['arch']['CuDNN']

        if "bidirectional" in self.config['arch']:
            bidire = self.config['arch']['bidirectional'][0]
            bidird = self.config['arch']['bidirectional'][1]
            bimerge = self.config['arch']['bimerge']
        else:
            bidire = False
            bidird = False
            bimerge = None

        if 'backwards' in self.config['arch']:
            backwards = self.config['arch']['backwards']
        else:
            backwards = False

        # GRU parameter for alternative implementation
        if 'after' in self.config['arch']:
            after = self.config['arch']['after']
        else:
            after = False

        if 'full' in self.config['arch']:
            full = self.config['arch']['full']
            fulldrop = self.config['arch']['fulldrop']
            activation_full = self.config['arch']['activation_full']
        else:
            full = []
            fulldrop = 0
            activation_full = 'linear'

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
            model = generate_recurrent_layer(neuronsE, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidire, bimerge, rseq=False)(input)
            model = generate_activation(activation)(model)

        else:
            model = generate_recurrent_layer(neuronsE, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidire, bimerge, rseq=True)(input)
            model = generate_activation(activation)(model)

            for i in range(1, nlayersE - 1):
                model = generate_recurrent_layer(neuronsE, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidire, bimerge, rseq=True)(model)
                model = generate_activation(activation)(model)

            model = generate_recurrent_layer(neuronsE, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidire, bimerge, rseq=False)(model)
            model = generate_activation(activation)(model)

        
        model = RepeatVector(odimensions)(model)

        decoder = model  # Keep decoder user for reusing it

        for i in range(nlayersD):
            model = generate_recurrent_layer(neuronsD, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidird, bimerge, rseq=True)(model)

            model = generate_activation(activation)(model)


        model = Concatenate()([decoder, model])
        for units in full:
            model = TimeDistributed(Dense(units=units))(model)
            model = TimeDistributed(generate_activation(activation_full))(model)
            model = TimeDistributed(Dropout(rate=fulldrop))(model)

        output = TimeDistributed(Dense(1))(model)
        self.model = Model(inputs=input, outputs=output)



