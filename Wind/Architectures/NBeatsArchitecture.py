"""
.. module:: NBeatsArchitecture

NBeatsArchitecture
******

:Description: NBeatsArchitecture

    Implementation of the NBeats architecture:

    Authors: Oreshkin, Boris N and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua
    Pub: International Conference on Learning Representations. 2019.
    Link: https://openreview.net/forum?id=r1ecqn4YwB


:Authors:
    bejar

:Version: 

:Date:  25/03/2021
"""

__author__ = 'bejar'


from keras.layers import Dense, Flatten, Dropout, Input, Add, Subtract
from keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation
from Wind.Train.Layers import generate_recurrent_layer

__author__ = 'bejar'


class NBeatsArchitecture(NNS2SArchitecture):
    """
    Recurrent architecture for sequence to sequence

    """
    modfile = None
    modname = 'NBeats'
    data_mode = ('2D', '2D')

    def main_block(self, input, neurons_input, neurons_forecast, neurons_backcast, activation, dropout):
        """
        The main block is composed by an input MLP and two linear transformations for forecasting and backcasting
        """
        # Originally four layers
        orig = Dense(neurons_input)(input)
        input_block = Dense(neurons_input)(input)
        input_block = generate_activation(activation)(input_block)
        for i in range(3):
            input_block = Dense(neurons_input)(input_block)
            input_block = generate_activation(activation)(input_block)
            input_block = Dropout(rate=dropout)(input_block)

        # Forecast output
        forecast = Dense(neurons_forecast)(input_block)
        forecast = generate_activation(activation)(forecast)
        forecast = Dropout(rate=dropout)(forecast)

        # Backcast output
        backcast = Dense(neurons_backcast)(input_block)
        backcast = generate_activation(activation)(backcast)
        backcast = Dropout(rate=dropout)(backcast)

        # We apply the subtraction to obtain the input for the next block
        return Subtract()([input, backcast]), forecast

    def group_block(self, n_blocks, input, neurons_input, neurons_forecast, neurons_backcast, activation, dropout):
        
        block, forecast_sum = self.main_block(input, neurons_input, neurons_forecast, neurons_backcast, activation, dropout)
        for i in range(n_blocks-1):
            block, forecast = self.main_block(block, neurons_input, neurons_forecast, neurons_backcast, activation, dropout)
            forecast_sum = Add()([forecast_sum, forecast])
            
        return block, forecast_sum
            

    def generate_model(self):
        """
        Model for NBeats architecture

        -------------
        json config:

          "arch": {
            "ninput": 64,
            "nforecast": 64,
            "nbackcast": 65,
              "niblocks": 3,
              "neblocks": 1,
             "dropout": 0.3,
            "activation": ["relu"],
            "mode":"NBeats"
          }

        :return:
        """
        neurons_input = self.config['arch']['ninput']
        neurons_forecast = self.config['arch']['nforecast']
        neurons_backcast= self.config['arch']['nbackcast']
        neurons_full= self.config['arch']['nfull']
        dropout = self.config['arch']['dropout']
        niblocks = self.config['arch']['niblocks']  # number of internal blocks
        neblocks = self.config['arch']['neblocks']  # number of external blocks
        activation = self.config['arch']['activation']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']
        impl = self.runconfig.impl

        input = Input(shape=(idimensions))
        eblock, forecast_sum = self.group_block(niblocks, input, neurons_input, neurons_forecast, neurons_backcast, activation, dropout)

        for i in range(neblocks-1):
            eblock, forecast = self.group_block(niblocks, eblock, neurons_input, neurons_forecast, neurons_backcast, activation, dropout)
            forecast_sum = Add()([forecast_sum, forecast])

        eforecast = Dense(neurons_full)(forecast_sum)
        eforecast = generate_activation(activation)(eforecast)

        output = Dense(odimensions, activation='linear')(eforecast)

        self.model = Model(inputs=input, outputs=output)
