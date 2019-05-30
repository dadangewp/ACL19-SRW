# -*- coding: utf-8 -*-


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, concatenate
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
    with codecs.open(fp, encoding="ANSI") as data_in:
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
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def buildEmbedding1 (X):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open('wiki.multi.en.vec', encoding='ANSI') as f:
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

def buildEmbedding2 (X):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open('wiki.multi.es.vec', encoding='ANSI') as f:
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
    
def RNN(X,X2):
    vocab1, embedding_matrix1 = buildEmbedding1(X)
    vocab2, embedding_matrix2 = buildEmbedding2(X2)
    en_inputs = Input(name='inputs',shape=(None,))
    es_inputs = Input(name='inputs2',shape=(None,))
    en_embedding = Embedding(vocab1+1, 300, input_length=100, weights=[embedding_matrix1])(en_inputs)
    es_embedding = Embedding(vocab2+1, 300, input_length=100, weights=[embedding_matrix2])(es_inputs)
    lstm1 = LSTM(16)(en_embedding)
    lstm2 = LSTM(16)(es_embedding)
    dense1 = Dense(4,name='FC1')(lstm1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(4,name='FC2')(lstm2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(0.4)(dense2)  
    concat = concatenate([dense1, dense2], axis=-1)
    layer = Dense(2,name='out_layer')(concat)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[en_inputs,es_inputs],outputs=layer)
    return model

dataTrain, dataLabel = parse_testing("es-en-ibereval-training.txt")
dataTrain2, dataLabel2 = parse_testing("es_ibereval_training.txt")

dataTest, labelTest = parse_testing("en_testing.tsv")  
dataTest2, labelTest2 = parse_testing("en-es-evalita-testing.txt") 

Y_train = pd.get_dummies(dataLabel)
Y_test = pd.get_dummies(labelTest)


max_len = 100
max_words = 15000

tok1 = Tokenizer(num_words=max_words)
tok1.fit_on_texts(dataTrain)
vocab1 = len(tok1.word_index)

tok2 = Tokenizer(num_words=max_words)
tok2.fit_on_texts(dataTrain2)
vocab2 = len(tok2.word_index)

sequences = tok1.texts_to_sequences(dataTrain)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

sequences2 = tok2.texts_to_sequences(dataTrain2)
sequences_matrix2 = sequence.pad_sequences(sequences2,maxlen=max_len)

test_sequences = tok1.texts_to_sequences(dataTest)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

test_sequences2 = tok2.texts_to_sequences(dataTest2)
test_sequences_matrix2 = sequence.pad_sequences(test_sequences2,maxlen=max_len)

allxtrain = [sequences_matrix,sequences_matrix2]

model = RNN(dataTrain, dataTrain2)
model.summary()
model.compile(loss='mse',optimizer=RMSprop(),metrics=[f1_score])
model.fit(allxtrain,Y_train,batch_size=16,epochs=4,
          validation_split=0.2)#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

allxtest = [test_sequences_matrix,test_sequences_matrix2]
accr = model.evaluate(allxtest, Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict(allxtest) 
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
