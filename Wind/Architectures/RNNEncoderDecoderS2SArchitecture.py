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
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Bidirectional, Dense, TimeDistributed, Flatten, RepeatVector
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


class RNNEncoderDecoderS2SArchitecture(NNS2SArchitecture):
    modfile = None
    modname = 'RNNEDS2S'
    data_mode = (False, '3D') #'s2s'

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

        if CuDNN:
            RNN = CuDNNLSTM if rnntype == 'LSTM' else CuDNNGRU
            self.model = Sequential()
            if nlayersE == 1:

                self.model.add(
                    RNN(neuronsE, input_shape=(idimensions), #kernel_initializer='zeros',
                        recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            else:
                self.model.add(RNN(neuronsE, input_shape=(idimensions), return_sequences=True,
                                   recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                for i in range(1, nlayersE - 1):
                    self.model.add(RNN(neuronsE, return_sequences=True, recurrent_regularizer=rec_regularizer,
                                       kernel_regularizer=k_regularizer))
                self.model.add(RNN(neuronsE, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

            self.model.add(RepeatVector(odimensions))

            for i in range(nlayersD):
                self.model.add(RNN(neuronsD, return_sequences=True, recurrent_regularizer=rec_regularizer,
                                   kernel_regularizer=k_regularizer))

            self.model.add(TimeDistributed(Dense(1)))
        else:
            RNN = LSTM if rnntype == 'LSTM' else GRU
            self.model = Sequential()
            if nlayersE == 1:
                self.model.add(RNN(neuronsE, input_shape=(idimensions), implementation=impl,
                                   recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                                   recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            else:
                self.model.add(RNN(neuronsE, input_shape=(idimensions), implementation=impl,
                                   recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                                   return_sequences=True, recurrent_regularizer=rec_regularizer,
                                   kernel_regularizer=k_regularizer))
                for i in range(1, nlayersE - 1):
                    self.model.add(RNN(neuronsE, recurrent_dropout=drop, implementation=impl,
                                       activation=activation, recurrent_activation=activation_r, return_sequences=True,
                                       recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                self.model.add(RNN(neuronsE, recurrent_dropout=drop, activation=activation,
                                   recurrent_activation=activation_r, implementation=impl,
                                   recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

            self.model.add(RepeatVector(odimensions))

            for i in range(nlayersD):
                self.model.add(RNN(neuronsD, recurrent_dropout=drop, implementation=impl,
                                   activation=activation, recurrent_activation=activation_r,
                                   return_sequences=True, recurrent_regularizer=rec_regularizer,
                                   kernel_regularizer=k_regularizer))

            self.model.add(TimeDistributed(Dense(1)))

    def evaluate(self, val_x, val_y, test_x, test_y):
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)
        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        if type(self.config['data']['ahead'])== list:
            ahead = self.config['data']['ahead'][1]
        else:
            ahead = self.config['data']['ahead']

        lresults = []
        for i in range(1, ahead + 1):
            lresults.append((i,
                             r2_score(val_y[:, i - 1], val_yp[:, i - 1]),
                             r2_score(test_y[:, i - 1], test_yp[:, i - 1])
                             ))
        return lresults

    # def summary(self):
    #     self.model.summary()
    #     neurons = self.config['arch']['neurons']
    #     neuronsD = self.config['arch']['neuronsD']
    #     nlayersE = self.config['arch']['nlayersE']  # >= 1
    #     nlayersD = self.config['arch']['nlayersD']  # >= 1
    #     activation = self.config['arch']['activation']
    #     activation_r = self.config['arch']['activation_r']
    #     print(f"lag: {self.config['data']['lag']} /Neurons: {neurons} {neuronsD} /Layers: ', {nlayersE} {nlayersD}"
    #           f"/Activation: {activation} {activation_r}")

    def log_result(self, result):
        for i, r2val, r2test in result:
            print(f"{self.config['arch']['mode']} |"
                  f" DNM= {self.config['data']['datanames'][0]},"
                  f" DS= {self.config['data']['dataset']},"
                  f" V= {len(self.config['data']['vars'])},"
                  f" LG= {self.config['data']['lag']},"
                  f" AH= {i},"
                  f" RNN= {self.config['arch']['rnn']},"
                  f" Bi={self.config['arch']['bimerge'] if self.config['arch']['bidirectional'] else 'no'},"
                  f" LY= {self.config['arch']['nlayersE']} {self.config['arch']['nlayersD']},"
                  f" NN= {self.config['arch']['neurons']} {self.config['arch']['neuronsD']},"
                  f" DR= {self.config['arch']['drop']:3.2f},"
                  f" AF= {self.config['arch']['activation']},"
                  f" RAF= {self.config['arch']['activation_r']},"
                  f" OPT= {self.config['training']['optimizer']},"
                  f" R2V = {r2val:3.5f}, R2T = {r2test:3.5f}")

            # print(
            #         '%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d %d, NN= %d %d, DR= %3.2f, AF= %s, RAF= %s, '
            #         'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
            #         (self.config['arch']['mode'],
            #          self.config['data']['datanames'][0],
            #          self.config['data']['dataset'],
            #          len(self.config['data']['vars']),
            #          self.config['data']['lag'],
            #          i,
            #          self.config['arch']['rnn'],
            #          self.config['arch']['bimerge'] if self.config['arch']['bidirectional'] else 'no',
            #          self.config['arch']['nlayersE'], self.config['arch']['nlayersD'],
            #          self.config['arch']['neurons'], self.config['arch']['neuronsD'],
            #          self.config['arch']['drop'],
            #          self.config['arch']['activation'],
            #          self.config['arch']['activation_r'],
            #          self.config['training']['optimizer'],
            #          r2val, r2test
            #          ))
