# -*- coding: utf-8 -*-


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, concatenate
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

def buildEmbedding (X):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open('hurtlex-embedding.vec', encoding='ANSI') as f:
        print("read embedding...")
        for line in f:
            values = line.split("\t")
            word = values[0]
            coefs = np.asarray(values[1].split(" "), dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    print("vocabulary size = "+str(len(tokenizer.word_index)))
    print("embedding size = "+str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 17))
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
            unk_embed = np.zeros(17, dtype='float32')
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]
    return vocab, embedding_matrix
    
def RNN(X):
    vocab, embedding_matrix = buildEmbedding(X)
    inputs = Input(name='inputs',shape=[max_len])
    embedding1 = Embedding(vocab+1, 32, input_length=100)(inputs)
    embedding2 = Embedding(vocab+1, 17, input_length=100, weights=[embedding_matrix], trainable=False)(inputs)
    lstm1 = LSTM(16)(embedding1)
    lstm2 = LSTM(2)(embedding2)
    #concat_in = Input(shape=(17,), dtype='float32')
    dense1 = Dense(8,name='FC1')(lstm1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(2,name='FC2')(lstm2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(0.4)(dense2)
    concat = concatenate([dense1, dense2], axis=-1)
    #dense3 = Dense(4,name='FC3')(concat)
    layer = Dense(2,name='out_layer')(concat)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
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

#hurtlexFeatures = np.concatenate((feature_an, feature_asf, feature_asm, feature_cds, feature_ddf, feature_ddp, feature_dmc, feature_is, feature_om, feature_or, feature_pa, feature_ps, feature_pr, feature_qas, feature_rci, feature_re, feature_svp), axis=0)

model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit(sequences_matrix,Y_train,batch_size=64,epochs=7)

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
