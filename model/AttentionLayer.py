# -*- coding: utf-8 -*-
# @auther tim
# @date 2016.11.30
from keras.engine import Layer
from keras.layers import Convolution1D, Permute, Multiply, Dense, Activation, K


class Sum(Layer):
    def call(self, inputs, **kwargs):
        return K.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class GAttenLayer(Layer):
    def __init__(self, **kwargs):
        super(GAttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3
        self.dense = Dense(1)
        self.trainable_weights = self.dense.trainable_weights
        super(GAttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        atten = self.dense(x)
        atten = Permute((2, 1))(atten)
        atten = Activation('softmax')(atten)
        atten = Permute((2, 1))(atten) #[n_sample, n_length, 1]
        conv_feature = Multiply()([x, atten]) #[n_sample, n_length, embedding]
        conv_feature = Sum()(conv_feature) #[n_sample, embedding]
        return conv_feature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class LAttenLayer(Layer):
    def __init__(self,nb_filter=1,filter_length=5,**kwargs):
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        super(LAttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3, 'attention layer 输入是三维矩阵'
        self.input_dim = input_shape[2]

        self.atten_layer = Convolution1D(
            filters=self.nb_filter,
            kernel_size=self.filter_length,
            padding='same',
            activation='sigmoid'
        )
        self.trainable_weights = self.atten_layer.trainable_weights
        super(LAttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        x_padding = x

        word_score = self.atten_layer(x_padding)
        out = Multiply()([x, word_score])
        return out

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)
