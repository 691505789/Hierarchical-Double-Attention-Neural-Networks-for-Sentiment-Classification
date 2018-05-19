# -*- coding: utf-8 -*-
# @auther tim
# @date 2016.11.30

import cPickle as cp
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Embedding, Input, Convolution1D, GlobalMaxPooling1D, Dropout, Concatenate, TimeDistributed, \
    Dense, \
    GRU, Bidirectional, Multiply

# 参数
from AttentionLayer import GAttenLayer
from Dataset import *
from common import ValTestLog, get_batch

EMBEDDING_DIM = 200 # 词向量维度
filter_lengths = [1,3,4,5] # CNN 卷积核大小
nb_filter = 100  # 卷积神经网络积极核函数的个数
fc_hidden_dims = 100 # 全连接层隐藏单元数量
batch_size = 32
MAX_SENT_LENGTH = {"IMDB":130, "yelp13":60, "yelp14":60}
gru_out_dim = 100
latten_range = 5


class data_loader(object):
    def __init__(self, dataname="IMDB"):
        self.dataname = dataname
    def load_datasets(self):
        print 'data loadeding....'
        # trainset = Dataset('../data/'+dataname+'/train.txt', voc)
        f = file('../data/'+self.dataname+'/trainset.save', 'rb')
        trainset = cp.load(f)
        f = file('../data/'+self.dataname+'/devset.save', 'rb')
        devset = cp.load(f)
        f = file('../data/'+self.dataname+'/testset.save', 'rb')
        testset = cp.load(f)
        f.close()

        print(trainset.docs[2].shape)
        print(trainset.label[2].shape)
        print(trainset.epoch)
        print(devset.epoch)
        print(testset.epoch)
        print 'data load finish...'
        return (trainset, devset, testset)

    def load_embedding_matrix(self):
        print 'word embedding matrix loading'
        f = file('../data/'+self.dataname+'/embinit.save', 'rb')
        embedding_matrix = cp.load(f)
        f.close()
        embedding_W = np.zeros_like(embedding_matrix)
        embedding_W[0] = embedding_matrix[-1]
        embedding_W[1:] = embedding_matrix[0:-1]
        print(embedding_W.shape)
        print 'word embedding matrix loading finish'
        return embedding_W

def atten(type='global', inputs=None, atten_range=None):
    assert type in ['global','local'], 'type in [global,local]'

    if type=='local' and atten_range==None:
        raise Exception('type is local, atten_range must be not None')

    # =============lstm 全局attention机制====================
    if type=='global' and inputs!=None:
        x_atten = GAttenLayer()(inputs)
        return x_atten

    # =============局部attention机制====================
    elif type=='local' and inputs!=None:
        x_score = Convolution1D(
            filters=1,
            kernel_size=atten_range,
            padding='same',
            activation='sigmoid'
        )(inputs)
        x_atten = Multiply()([x_score, inputs])
        # x_atten = LAttenLayer(filter_length=atten_range)(inputs)
        return x_atten

def lstm_atten(sent_sequences, out_dim):
    sent_sequences = Bidirectional(GRU(out_dim, return_sequences=True))(sent_sequences)
    #sent_sequences = TimeDistributed(Dense(2*out_dim))(sent_sequences)
    # =============lstm 全局attention机制====================
    doc_presentation = atten(type="global", inputs=sent_sequences)
    return doc_presentation

def conv_atten(sent_sequences, nb_filter, filter_lengths, atten_range):
    # ================局部注意力层===================
    # local attention层开始
    sent_sequences = atten(type='local', atten_range=atten_range,inputs=sent_sequences)

    # 卷积神经网络提取特征
    doc_features = []
    for filter_length in filter_lengths:
        conv_out = Convolution1D(
            filters=nb_filter,
            kernel_size=filter_length,
            padding='same',
            activation='relu'
        )(sent_sequences)
        pooling_out = GlobalMaxPooling1D()(conv_out)
        # pooling_out = Dropout(0.5)(pooling_out)
        doc_features.append(pooling_out)


    doc_representation = Concatenate()(doc_features)
    doc_representation = Dropout(0.5)(doc_representation)
    return doc_representation

def get_model(
    embedding_W,
    dataname,
    model_type="cnn2cnn",
):
    assert model_type in ['cnn2cnn', 'cnn2rnn'], 'type in [cnn2cnn,cnn2rnn]'
    #================sentence========================
    sentence_input = Input(shape=(MAX_SENT_LENGTH.get(dataname),), dtype='int32')
    # word embedding 层
    embedding_layer = Embedding(embedding_W.shape[0],
                                EMBEDDING_DIM,
                                weights=[embedding_W],
                                trainable=True)
    embedding_sequences = embedding_layer(sentence_input)
    sents_representation = conv_atten(embedding_sequences, nb_filter, filter_lengths,latten_range)
    sent_model = Model(sentence_input, sents_representation, name='sentModel')

    # ==============================document================
    doc_input = Input(shape=(None, MAX_SENT_LENGTH.get(dataname)), dtype='int32')
    sent_sequences = TimeDistributed(sent_model)(doc_input)

    if model_type=='cnn2cnn':
        doc_representation = conv_atten(sent_sequences,nb_filter,filter_lengths,atten_range=latten_range)
    elif model_type=='cnn2rnn':
        doc_representation = lstm_atten(sent_sequences, out_dim=gru_out_dim)
    fc_out = Dense(units=fc_hidden_dims, activation='relu',
                   name='fcLayer')(doc_representation)
    fc_out = Dropout(0.5)(fc_out)
    preds = Dense(classes, activation='softmax')(fc_out)
    model = Model(doc_input, preds)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    return model

if __name__=="__main__":
    model_type = sys.argv[1]
    dataname = sys.argv[2]
    classes = int(sys.argv[3])

    dataLoader = data_loader(dataname=dataname)
    trainset, devset, testset = dataLoader.load_datasets()
    embedding_W = dataLoader.load_embedding_matrix()


    checkpointer = ModelCheckpoint(
        filepath="../save/weights.hdf5",
        verbose=1,
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=False
    )
    testCallback = ValTestLog(dataset=testset, classes=classes)

    for _ in range(10):
        model = get_model(embedding_W=embedding_W, dataname=dataname, model_type=model_type)
        print model.summary()
        model.fit_generator(
            get_batch(dataset=trainset, classes=classes),
            steps_per_epoch=trainset.epoch,
            epochs=15,
            verbose=2,
            validation_data=get_batch(dataset=devset, classes=classes),
            validation_steps=devset.epoch,
            callbacks=[testCallback]
        )

