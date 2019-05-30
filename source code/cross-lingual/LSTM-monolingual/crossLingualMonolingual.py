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
from keras_self_attention import SeqSelfAttention

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) 

def parse_training(fp):
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[0])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def buildEmbedding (X):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open('cc.en.300.vec', encoding='ANSI') as f:
        print("read embedding...")
        for line in f:
            values = line.split(" ")
            word = values[0]
            #print(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    print("vovabulary size = "+str(len(tokenizer.word_index)))
    print("embedding size = "+str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    unk_dict = {}
    vocab = len(tokenizer.word_index)
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif word in unk_dict:
            embedding_matrix[i] = unk_dict[word]
        else:
            # random init, see https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
            unk_embed = np.random.random(300) * -2 + 1
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]
    return vocab, embedding_matrix
    
def RNN(X):
    vocab, embedding_matrix = buildEmbedding(X)
    inputs = Input(name='inputs',shape=(None,))
    layer = Embedding(vocab+1, 300, input_length=100, weights=[embedding_matrix])(inputs)
    layer = LSTM(16)(layer)
    layer = Dense(4,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

dataTrain, dataLabel = parse_training("harassment-training.txt")
dataTest, labelTest = parse_training("wassem-testing.txt")  

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
model.compile(loss='mse',optimizer=RMSprop(),metrics=['acc'])
model.fit(sequences_matrix,Y_train,batch_size=16,epochs=3)

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
#print(avg)
print(prec)
print(rec)
