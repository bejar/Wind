"""
.. module:: MLPS2SRecursiveArchirecture

MLPS2SRecursiveArchirecture
*************

:Description: MLPS2SRecursiveArchirecture

    

:Authors: bejar
    

:Version: 

:Created on: 30/11/2018 13:20 

"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Input, concatenate, Flatten

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True


__author__ = 'bejar'

class MLPS2SRecursiveArchitecture(NNS2SArchitecture):
    modfile = None
    modname = 'MLPS2SREC'
    data_mode = ('2D', '2D')  #'mlp'

    def generate_model(self):
        """
        Model for MLP recursive multiple regression (s2s)

        It takes as inputs the data and the predictions of the previous step

        :return:
        """

        activation = self.config['arch']['activation']
        dropout = self.config['arch']['drop']
        full_layers = self.config['arch']['full']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']
        rdimensions = self.config['rdimensions']

        input = Input(shape=(idimensions))
        finput = Flatten(input)
        if rdimensions > 0:
            rinput = Input(shape=(rdimensions))
            recinput = concatenate([finput, rinput])
        else:
            recinput = finput

        output = Dense(odimensions, activation='linear')
        model = Dense(full_layers[0], input_shape=idimensions, activation=activation)(recinput)
        model = Dropout(rate=dropout)(model)

        for units in full_layers[1:]:
            model = Dense(units=units, activation=activation)(model)
            model = Dropout(rate=dropout)(model)

        if rdimensions > 0:
            self.model = Model(inputs=[input, rinput], outputs=output)
        else:
            self.model = Model(inputs=input, outputs=output)


    def summary(self):
        self.model.summary()
        activation = self.config['arch']['activation']
        print(
        f"lag: {self.config['data']['lag']} /Layers: {str(self.config['arch']['full'])} /Activation: {activation}")

        print()

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
                  f"AF={self.config['arch']['activation']},"             
                  f"OPT={self.config['training']['optimizer']},"
                  f"R2V={r2val:3.5f},"
                  f"R2T={r2test:3.5f}"
                  )

    def predict(self, val_x):
        """
        Returns the predictions of the model for some data

        :param val_x:
        :param val_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)

        return self.model.predict(val_x, batch_size=batch_size, verbose=0)
