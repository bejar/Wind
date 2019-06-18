"""
.. module:: SelfAttention

SelfAttention
*************

:Description: SelfAttention

    

:Authors: Borrowed from Marc Nuth Github https://github.com/Marcnuth/keras-attention
    

:Version: 

:Created on: 17/06/2019 13:46 

"""

__author__ = 'bejar'


from keras.layers import Layer
from keras import initializers
from keras import backend as K


class SelfAttention(Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.supports_masking = True

    def build(self, input_shape):
        self.context = self.add_weight(name='context',
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                       regularizer=self.regularizer,
                                       trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
        attention = attention_in / K.expand_dims(K.sum(attention_in, axis=-1), -1)

        if mask is not None:
            attention = attention * K.cast(mask, 'float32')

        weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, input, input_mask=None):
        return None