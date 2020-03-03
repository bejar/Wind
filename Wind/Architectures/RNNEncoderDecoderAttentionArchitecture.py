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
from Wind.Util.AttentionDecoder import AttentionDecoder
from keras.models import load_model, Model
from keras.layers import LSTM, GRU, Dense, TimeDistributed, Input
from keras.layers import Activation, dot, concatenate, Permute
import numpy as np
from Wind.ErrorMeasure import ErrorMeasure
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


class RNNEncoderDecoderAttentionArchitecture(NNS2SArchitecture):
    """
    Recurrent encoder decoder with simple attention
    """
    modfile = None
    modname = 'RNNEDATT'
    data_mode = ('3D', '3D')  # 's2s'

    def generate_model(self):
        """
        Model for RNN with Encoder Decoder for S2S with attention
        -------------
        json config:

        "arch": {
            "neurons":32,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0.3,
            "nlayersE": 1,
            "nlayersD": 1,
            "activation": "relu",
            "activation_r": "hard_sigmoid",
            "CuDNN": false,
            "rnn": "GRU",
            "full": [64, 32],
            "mode": "RNN_ED_s2s_att"
        }

        :return:
        """
        neurons = self.config['arch']['neurons']
        attsize = self.config['arch']['attsize']
        drop = self.config['arch']['drop']
        nlayersE = self.config['arch']['nlayersE']  # >= 1

        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        activation_fl = self.config['arch']['activation_fl']
        rec_reg = self.config['arch']['rec_reg']
        rec_regw = self.config['arch']['rec_regw']
        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']
        rnntype = self.config['arch']['rnn']
        CuDNN = self.config['arch']['CuDNN']
        # neuronsD = self.config['arch']['neuronsD']
        full_layers = self.config['arch']['full']

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

        # Encoder RNN - First Input
        enc_input = Input(shape=(idimensions))
        encoder = RNN(neurons, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      recurrent_regularizer=rec_regularizer, return_sequences=True, kernel_regularizer=k_regularizer)(
            enc_input)

        for i in range(1, nlayersE):
            encoder = RNN(neurons, implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, return_sequences=True,
                          kernel_regularizer=k_regularizer)(
                encoder)

        decoder = AttentionDecoder(attsize, odimensions)(encoder)
        decoder = Permute((2,1))(decoder)

        output = TimeDistributed(Dense(full_layers[0], activation=activation_fl))(decoder)
        for l in full_layers[1:]:
            output = TimeDistributed(Dense(l, activation=activation_fl))(output)

        output = TimeDistributed(Dense(1, activation="linear"))(output)


        self.model = Model(inputs=enc_input, outputs=output)

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        The evaluation for this architecture is iterative, for each step a new time in the future is predicted
        using the results of the previous steps, the result is appended for the next step

        :param save_errors:
        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile, custom_objects={"AttentionDecoder": AttentionDecoder})
            # self.model = load_model(self.modfile)

        # if type(self.config['data']['ahead']) == list:
        #     ahead = self.config['data']['ahead'][1]
        # else:
        #     ahead = self.config['data']['ahead']

        # Maintained to be compatible with old configuration files
        if type(self.config['data']['ahead'])==list:
            iahead = self.config['data']['ahead'][0]
            ahead = (self.config['data']['ahead'][1] - self.config['data']['ahead'][0]) + 1
        else:
            iahead = 1
            ahead = self.config['data']['ahead']

        # The input for the first step is the last column of the training (t-1)
        val_x_tfi = val_x[0][:, -1, 0]
        val_x_tfi = val_x_tfi.reshape(val_x_tfi.shape[0], 1, 1)
        test_x_tfi = test_x[0][:, -1, 0]
        test_x_tfi = test_x_tfi.reshape(test_x_tfi.shape[0], 1, 1)

        # Copy the first slice (time step 1)
        val_x_tf = val_x_tfi.copy()
        test_x_tf = test_x_tfi.copy()

        lresults = []
        for i in range(1, ahead + 1):
            val_yp = self.model.predict([val_x[0], val_x_tf], batch_size=batch_size, verbose=0)
            test_yp = self.model.predict([test_x[0], test_x_tf], batch_size=batch_size, verbose=0)

            val_x_tf = np.concatenate((val_x_tfi, val_yp), axis=1)
            test_x_tf = np.concatenate((test_x_tfi, test_yp), axis=1)

        if save_errors is not None:
            f = h5py.File(f'errors{self.modname}-S{self.config["data"]["datanames"][0]}{save_errors}.hdf5', 'w')
            dgroup = f.create_group('errors')
            dgroup.create_dataset('val_y', val_y.shape, dtype='f', data=val_y, compression='gzip')
            dgroup.create_dataset('val_yp', val_yp.shape, dtype='f', data=val_yp, compression='gzip')
            dgroup.create_dataset('test_y', test_y.shape, dtype='f', data=test_y, compression='gzip')
            dgroup.create_dataset('test_yp', test_yp.shape, dtype='f', data=test_y, compression='gzip')
            if scaler is not None:
                # n-dimensional vectors
                dgroup.create_dataset('val_yu', val_y.shape, dtype='f', data=scaler.inverse_transform(val_y), compression='gzip')
                dgroup.create_dataset('val_ypu', val_yp.shape, dtype='f', data=scaler.inverse_transform(val_yp), compression='gzip')
                dgroup.create_dataset('test_yu', test_y.shape, dtype='f', data=scaler.inverse_transform(test_y), compression='gzip')
                dgroup.create_dataset('test_ypu', test_yp.shape, dtype='f', data=scaler.inverse_transform(test_yp), compression='gzip')


        # After the loop we have all the predictions for the ahead range
        for i, p in zip(range(1, ahead + 1), range(iahead, self.config['data']['ahead'][1]+1)):
            lresults.append([p]  + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                               val_yp[:, i - 1],
                                                               test_y[:, i - 1],
                                                               test_yp[:, i - 1]))


        return lresults

