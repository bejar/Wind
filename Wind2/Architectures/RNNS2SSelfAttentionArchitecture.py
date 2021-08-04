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

from Wind2.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import load_model, Model
from keras.layers import LSTM, GRU, Dense,  Dropout, Input
from Wind.Train.Activations import generate_activation

from Wind.Util.SelfAttention import SelfAttention
from Wind2.ErrorMeasure import ErrorMeasure
import h5py

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


class RNNS2SSelfAttentionArchitecture(NNS2SArchitecture):
    """
    Recurrent architecture for sequence to sequence

    """
    modfile = None
    modname = 'RNNS2SATT'
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
        attsize = self.config['arch']['attsize']

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

        input = Input(shape=(idimensions))

        # self.model = Sequential()
        if nlayersE == 1:
            model = RNN(neurons,
                        implementation=impl, return_sequences=True,
                        recurrent_dropout=drop,
                        recurrent_activation=activation_r,
                        recurrent_regularizer=rec_regularizer,
                        kernel_regularizer=k_regularizer)(input)
            model = generate_activation(activation)(model)
        else:
            model = RNN(neurons,
                        implementation=impl, return_sequences=True,
                        recurrent_dropout=drop,
                        recurrent_activation=activation_r,
                        recurrent_regularizer=rec_regularizer,
                        kernel_regularizer=k_regularizer)(input)
            model = generate_activation(activation)(model)

            for i in range(1, nlayersE - 1):
                model = RNN(neurons,
                            implementation=impl, return_sequences=True,
                            recurrent_dropout=drop,
                            recurrent_activation=activation_r,
                            recurrent_regularizer=rec_regularizer,
                            kernel_regularizer=k_regularizer)(model)
                model = generate_activation(activation)(model)


            model = RNN(neurons,
                        implementation=impl, return_sequences=True,
                        recurrent_dropout=drop,
                        recurrent_activation=activation_r,
                        recurrent_regularizer=rec_regularizer,
                        kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)

        ## OLD self attention code
        # attention = TimeDistributed(Dense(1, activation='tanh'))(model)
        # attention = Flatten()(attention)
        # attention = Activation('softmax')(attention)
        # attention = RepeatVector(neurons)(attention)
        # attention = Permute([2,1])(attention)
        #
        # sent_representation = multiply([model, attention])
        # sent_representation = Lambda(lambda xin: K.sum(xin, axis=-1))(sent_representation)
        #
        # # reg = TimeDistributed(Dense(1, activation='linear'))(sent_representation)
        # model = Dense(1, activation='linear')(sent_representation)
        # # model = Flatten()(reg)

        model = SelfAttention(attention_type= 'additive')(model)
        for nn in full:
            model = Dense(nn)(model)
            model = generate_activation(activation_full)(model)
            model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)
        # self.model = Model(inputs=input, outputs=reg)

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        Evaluates the trained model with validation and test
        this function uses a custom object

        Overrides parent function

        :param save_errors:
        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
                self.model = load_model(self.modfile, custom_objects={"SelfAttention":SelfAttention})
#                self.model = load_model(self.modfile)
        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        # Maintained to be compatible with old configuration files
        if type(self.config['data']['ahead'])==list:
            iahead = self.config['data']['ahead'][0]
            ahead = (self.config['data']['ahead'][1] - self.config['data']['ahead'][0]) + 1
        else:
            iahead = 1
            ahead = self.config['data']['ahead']

        if save_errors is not None:
            f = h5py.File(f'errors{self.modname}-S{self.config["data"]["datanames"][0]}{save_errors}.hdf5', 'w')
            dgroup = f.create_group('errors')
            dgroup.create_dataset('val_y', val_y.shape, dtype='f', data=val_y, compression='gzip')
            dgroup.create_dataset('val_yp', val_yp.shape, dtype='f', data=val_yp, compression='gzip')
            dgroup.create_dataset('test_y', test_y.shape, dtype='f', data=test_y, compression='gzip')
            dgroup.create_dataset('test_yp', test_yp.shape, dtype='f', data=test_y, compression='gzip')

        lresults = []

        for i, p in zip(range(1, ahead + 1), range(iahead, self.config['data']['ahead'][1]+1)):
            lresults.append([p]  + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                               val_yp[:, i - 1],
                                                               test_y[:, i - 1],
                                                               test_yp[:, i - 1],scaler=scaler))
        return lresults


