# -*- coding: utf-8 -*-

import codecs

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
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

def parse_training_cast(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[0]
                label = line.split("\t")[1]
                if "OFFENSE" in label :
                    misogyny = 1
                else :
                    misogyny = 0
                #misogyny = int(line.split("\t")[0])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
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

def parse_testing_cast(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                label = line.split("\t")[2]
                if "OFF" in label :
                    misogyny = 1
                else :
                    misogyny = 0
                #misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_label(fp):
    label = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                y = int(line)
                #if y == 0:
                 #   x = "not irony"
                #elif y == 1:
                 #   x = "polarity contrast"
                #elif y == 2 :
                 #   x = "other irony"
                #else :
                 #   x = "situational irony"
                label.append(y)

    return label

def parse_gold(fp):
    label = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                y = int(line.split("\t")[1])
                #if y == 0:
                 #   x = "not irony"
                #elif y == 1:
                 #   x = "polarity contrast"
                #elif y == 2 :
                 #   x = "other irony"
                #else :
                 #   x = "situational irony"
                label.append(y)

    return label