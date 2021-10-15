"""
.. module:: CNNS2S2DSeparateArchitecture

CNNS2SArchitecture
*************


:Description: CNNS2S2DSeparateArchitecture

    Class for convolutional sequence to sequence architecture 
    with separate branch for neighbors and 2D convolutions


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from tensorflow.keras.layers import Dense, Dropout, SeparableConv1D, Flatten, Conv2D, SeparableConv2D
from tensorflow.keras.layers import  Input, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class CNNSeparableS2S2DSeparateArchitecture(NNS2SArchitecture):
    """
    Class for separable convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNS2S'
    data_mode = ('3D', '2D')  # 'cnn'

    def generate_model(self):
        """
        Model for separable CNN for S2S

        json config:

        "arch": {
            "filters": [32],
            "strides": [1],
            "dilation": false,
            "kernel_size": [3],
            "depth_multiplier": 1,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0,
            "activation": "relu",
            "activation_full": "linear",
            "full": [16,8],
            "fulldrop": 0,
            "mode":"CNN_sep_s2s"
        }

        :return:
        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        padding = self.config['arch']['padding']
        kernel_size = self.config['arch']['kernel_size']
        depth_multiplier = self.config['arch']['depth_multiplier']

        dropa = self.config['arch']['dropa']
        filtersa = self.config['arch']['filtersa']
        kernel_sizea = self.config['arch']['kernel_sizea']
        depth_multipliera = self.config['arch']['depth_multipliera']

        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation = self.config['arch']['strides']
            strides = [1] * len(dilation)
        else:
            strides = self.config['arch']['strides']
            dilation = [1] * len(strides)

        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilationa' in self.config['arch'] and self.config['arch']['dilationa']:
            dilationa = self.config['arch']['stridesa']
            stridesa = [1] * len(dilationa)
        else:
            stridesa = self.config['arch']['stridesa']
            dilationa = [1] * len(stridesa)


        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']

        activation = self.config['arch']['activation']

        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        if 'batchnorm' in self.config['arch']:
            bnorm = self.config['arch']['batchnorm']
        else:
            bnorm = False

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif k_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        # Principal branch
        input = Input(shape=(idimensions[0]))
        model = SeparableConv1D(filters[0], input_shape=(idimensions[0]), kernel_size=kernel_size[0], strides=strides[0],
                                padding=padding, dilation_rate=dilation[0], depth_multiplier=depth_multiplier,
                                kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)
        if bnorm:
            model = BatchNormalization()(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        for i in range(1, len(filters)):
            if len(filters) > len(strides):
                model = SeparableConv1D(filters[i], kernel_size=kernel_size[0], strides=strides[0],
                                        padding=padding, dilation_rate=dilation[0], depth_multiplier=depth_multiplier,
                                        kernel_regularizer=k_regularizer)(model)
            else:
                model = SeparableConv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                                        padding=padding, dilation_rate=dilation[i], depth_multiplier=depth_multiplier,
                                        kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)
            if bnorm:
                model = BatchNormalization()(model)

            if drop != 0:
                model = Dropout(rate=drop)(model)

        # Additional branch
        # The input is examples x lag x variables x sites
        inputa = Input(shape=(idimensions[1]))


        # We follow with 1xnvars convolutions and then convolutions in the temporal dimension
        vardim = idimensions[1][1]

        modela = SeparableConv2D(filtersa[0], kernel_size=(1,vardim), strides=(1,1),
                                padding=padding, dilation_rate=dilationa[0],depth_multiplier=depth_multipliera,
                                kernel_regularizer=k_regularizer)(input)

        modela = generate_activation(activation)(modela)
        if bnorm:
            modela = BatchNormalization()(modela)

        if drop != 0:
            modela = Dropout(rate=dropa)(modela)

        modela = SeparableConv2D(filtersa[0], kernel_size=(kernel_sizea[0],1), strides=(stridesa[0],1),
                                padding=padding, dilation_rate=dilationa[0], depth_multiplier=depth_multipliera,
                                kernel_regularizer=k_regularizer)(modela)

        modela = generate_activation(activation)(modela)
        if bnorm:
            modela = BatchNormalization()(modela)

        if drop != 0:
            modela = Dropout(rate=dropa)(modela)

        # First a 1x1 2D convolution to mix all the sites
        modela =  Conv2D(filters[1], kernel_size=(1,1), strides=stridesa[0],
            padding=padding, dilation_rate=dilationa[0], kernel_regularizer=k_regularizer)(modela)
        modela = generate_activation(activation)(modela)
        if bnorm:
            modela = BatchNormalization()(modela)

        if drop != 0:
            modela = Dropout(rate=dropa)(modela)

        # Fusion
        model = Flatten()(model)
        modela = Flatten()(modela)

        modelf = concatenate([model, modela])
        for l in full_layers:
            modelf = Dense(l)(modelf)
            modelf = generate_activation(activationfl)(modelf)
            if bnorm:
                modelf = BatchNormalization()(modelf)
            if fulldrop != 0:
                modelf = Dropout(rate=fulldrop)(modelf)

        output = Dense(odimensions, activation='linear')(modelf)

        self.model = Model(inputs=[input,inputa], outputs=output)



        # # Additional branch
        # # The input is examples x lag x variables x sites
        # inputa = Input(shape=(idimensions[1]))
        # # First a 1x1 2D convolution to mix all the sites
        # modela =  Conv2D(filters[0], input_shape=(idimensions[1]),kernel_size=(1,1), strides=stridesa[0],
        #     padding=padding, dilation_rate=dilationa[0], kernel_regularizer=k_regularizer)(inputa)
        # modela = generate_activation(activation)(modela)
        # if bnorm:
        #     modela = BatchNormalization()(modela)

        # if drop != 0:
        #     modela = Dropout(rate=dropa)(modela)

        # # We follow with 1xnvars convolutions and then convolutions in the temporal dimension
        # vardim = idimensions[1][1]
        # for i in range(1, len(filtersa)):
        #     modela = SeparableConv2D(filtersa[i], kernel_size=(1,vardim), strides=(1,1),
        #                             padding=padding, dilation_rate=dilationa[0],depth_multiplier=depth_multipliera,
        #                             kernel_regularizer=k_regularizer)(modela)
   
        #     modela = generate_activation(activation)(modela)
        #     if bnorm:
        #         modela = BatchNormalization()(modela)

        #     if drop != 0:
        #         modela = Dropout(rate=dropa)(modela)

        #     modela = SeparableConv2D(filtersa[i], kernel_size=(kernel_sizea[0],1), strides=(stridesa[0],1),
        #                             padding=padding, dilation_rate=dilationa[0], depth_multiplier=depth_multipliera,
        #                             kernel_regularizer=k_regularizer)(modela)
   
        #     modela = generate_activation(activation)(modela)
        #     if bnorm:
        #         modela = BatchNormalization()(modela)

        #     if drop != 0:
        #         modela = Dropout(rate=dropa)(modela)
