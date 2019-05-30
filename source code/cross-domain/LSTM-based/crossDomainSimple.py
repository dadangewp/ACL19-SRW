# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) 

def parse_training(fp):
    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[0])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing_cast(fp):
    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                label = line.split("\t")[2]
                if "OFF" in label:
                    misogyny = 1
                else :
                    misogyny = 0
                #misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):
    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y
    
def RNN(X):
    inputs = Input(name='inputs',shape=(None,))
    layer = Embedding(vocab+1, 32, input_length=100)(inputs)
    layer = LSTM(16)(layer)
    layer = Dense(8,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

dataTrain, dataLabel = parse_testing_cast("olid-training-v1.0.tsv")
dataTest, labelTest = parse_testing_cast("testset-levela.tsv")  

Y_train = pd.get_dummies(dataLabel)
Y_test = pd.get_dummies(labelTest)

max_len = 100
max_words = 15000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain)
vocab = len(tok.word_index)

sequences = tok.texts_to_sequences(dataTrain)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(dataTest)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit(sequences_matrix,Y_train,batch_size=64,epochs=5)

accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict(test_sequences_matrix) 
y_pred = np.argmax(y_prob, axis=-1)
acc = metrics.accuracy_score(Y_test[1], y_pred) 
score_pos = metrics.f1_score(Y_test[1], y_pred, pos_label=1)
score_neg = metrics.f1_score(Y_test[1], y_pred, pos_label=0)
prec = metrics.precision_score(Y_test[1], y_pred, pos_label=1)
rec = metrics.recall_score(Y_test[1], y_pred, pos_label=1)
tn, fp, fn, tp = confusion_matrix(Y_test[1],y_pred).ravel()
avg = (score_pos + score_neg)/2
print(acc)
print(score_pos)
print(avg)
print(prec)
print(rec)
