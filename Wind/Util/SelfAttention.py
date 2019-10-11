"""
.. module:: SelfAttention

SelfAttention
*************

:Description: SelfAttention

:Authors: Borrowed from Marc Nuth Github https://github.com/Marcnuth/keras-attention
          and Zhao HG https://github.com/CyberZHG/keras-self-attention
          
          additive attention from https://arxiv.org/pdf/1806.01264.pdf
    
:Version: 0.1

:Created on: 20/09/2019 13:12

"""

__author__ = 'bejar'


from keras.layers import Layer
from keras import initializers
from keras import backend as K


class SelfAttention(Layer):
    
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'
    
    def __init__(self,  
                       attention_type=ATTENTION_TYPE_ADD, **kwargs):
        
        super(SelfAttention, self).__init__(**kwargs)
        
        self.regularizer = None
        self.supports_masking = True
        self.attention_type = attention_type
        
        if attention_type == SelfAttention.ATTENTION_TYPE_ADD:
            self.Weight_t, self.Weight_x = None, None
        elif attention_type == SelfAttention.ATTENTION_TYPE_MUL:
            self.context = None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)
            
        
    def get_config(self):
        config = {
            'attention_type': self.attention_type,
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#        return dict(list(base_config.items()))

    def build(self, input_shape):
        
        if self.attention_type == SelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive(input_shape)
        elif self.attention_type == SelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative(input_shape)
            
        super(SelfAttention, self).build(input_shape) 
        return None
        
    def _build_additive(self, input_shape):
                        
        self.Weight_t = self.add_weight(name='Weight_t',
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                       regularizer=self.regularizer,
                                       trainable=True)
        self.Weight_x = self.add_weight(name='Weight_x',
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                       regularizer=self.regularizer,
                                       trainable=True)
        return None

    def _build_multiplicative(self, input_shape):
        
        self.context = self.add_weight(name='context',
                                      shape=(input_shape[-1], 1),
                                      initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                      regularizer=self.regularizer,
                                      trainable=True)
        return None
        
    def call(self, x, mask=None):
        
        if self.attention_type == SelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive(x)
        elif self.attention_type == SelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative(x)
            
        attention_top = K.softmax(e)
        attention = attention_top /K.expand_dims(K.sum(attention_top, axis=-1), -1)
        weigthed_end = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        if mask is not None:
            attention = attention * K.cast(mask, 'float32')
        return weigthed_end
    
    def _call_additive(self, x):
       
        b = K.squeeze(K.dot(x, self.Weight_x), axis=-1)
        a = K.squeeze(K.dot(x, self.Weight_t), axis=-1)
        
        return(a+b)
        
    def _call_multiplicative(self, x):
       
       return (K.squeeze(K.dot(x, self.context), axis=-1))
          
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, input, input_mask=None):
        return None
    
    @staticmethod
    def get_custom_objects():
        return {'SelfAttention': SelfAttention}
