"""
.. module:: CNNS2SArchitecture

CNNS2SArchitecture
*************


:Description: CNNS2SArchitecture

    Class for convolutional sequence to sequence architecture


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.metrics import r2_score

from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


class CNNS2SArchitecture(NNS2SArchitecture):
    """
    Class for convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNS2S'
    data_mode = (False, '2D') #'cnn'

    def generate_model(self):
        """
        Model for CNN with Encoder Decoder for S2S

        json config:

        "arch": {
            "filters": [32],
            "strides": [1],
            "kernel_size": [3],
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0,
            "nlayers": 1,
            "activation": "relu",
            "activationfl": "linear",
            "full": [16,8],
            "mode":"CNN_s2s"
        }

        :return:
        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        kernel_size = self.config['arch']['kernel_size']
        strides = self.config['arch']['strides']
        activationfl = self.config['arch']['activationfl']
        full_layers = self.config['arch']['full']

        activation = self.config['arch']['activation']

        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif k_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        self.model = Sequential()

        self.model.add(Conv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                              activation=activation, padding='causal',
                              kernel_regularizer=k_regularizer))

        if drop != 0:
            self.model.add(Dropout(rate=drop))

        for i in range(1, len(filters)):
            self.model.add(Conv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                                  activation=activation, padding='causal',
                                  kernel_regularizer=k_regularizer))
            if drop != 0:
                self.model.add(Dropout(rate=drop))

        self.model.add(Flatten())
        for l in full_layers:
            self.model.add(Dense(l, activation=activationfl))

        self.model.add(Dense(odimensions, activation='linear'))

    # def summary(self):
    #     self.model.summary()
    #     print(f"LAG={self.config['data']['lag']} STRIDES={self.config['arch']['strides']} "
    #           f"KER_S={self.config['arch']['kernel_size']} FILT={self.config['arch']['filters']} "
    #           f"DROP={self.config['arch']['drop']}")
    #
    #     print()

    def evaluate(self, val_x, val_y, test_x, test_y):
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)
        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        # Maintained to be compatible with old configuration files
        if type(self.config['data']['ahead'])==list:
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

    def log_result(self, result):
        for i, r2val, r2test in result:
            print(f"{self.config['arch']['mode']} |"
                  f"DNM={self.config['data']['datanames'][0]},"
                  f"DS={self.config['data']['dataset']},"
                  f"V={len(self.config['data']['vars'])},"
                  f"LG={self.config['data']['lag']},"
                  f"AH={i},"
                  f"FL={str(self.config['arch']['full'])},"
                  f"DR={self.config['arch']['drop']},"
                  f"AF={self.config['arch']['activationfl']},"
                  f"ACNN={self.config['arch']['activation']},"
                  f"FIL={str(self.config['arch']['filters'])},"
                  f"KS={str(self.config['arch']['kernel_size'])},"
                  f"STR={str(self.config['arch']['strides'])},"                  
                  f"OPT={self.config['training']['optimizer']},"
                  f"R2V={r2val:3.5f},"
                  f"R2T={r2test:3.5f}"
                  )


            # print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
            #       'ACNN= %s, FIL=%s, KS=%s, STR=%s, OPT= %s, R2V = %3.5f, R2T = %3.5f' %
            #       (self.config['arch']['mode'],
            #        self.config['data']['datanames'][0],
            #        self.config['data']['dataset'],
            #        len(self.config['data']['vars']),
            #        self.config['data']['lag'],
            #        i, str(self.config['arch']['full']),
            #        self.config['arch']['drop'],
            #        self.config['arch']['activationfl'],
            #        self.config['arch']['activation'],
            #        str(self.config['arch']['filters']),
            #        str(self.config['arch']['kernel_size']),
            #        str(self.config['arch']['strides']),
            #        self.config['training']['optimizer'],
            #        r2val,
            #        r2test,
            #        ))
