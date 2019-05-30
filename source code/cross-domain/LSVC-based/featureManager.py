# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
import requests
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix, hstack
from emojiExtractor import Emoji
from emojiSentiment import EmojiSentiment
from swearWordExtractor import Swear
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lexicon_en.en_an import EN_AN
from lexicon_en.en_asf import EN_ASF
from lexicon_en.en_asm import EN_ASM
from lexicon_en.en_cds import EN_CDS
from lexicon_en.en_ddf import EN_DDF
from lexicon_en.en_ddp import EN_DDP
from lexicon_en.en_dmc import EN_DMC
from lexicon_en.en_is import EN_IS
from lexicon_en.en_om import EN_OM
from lexicon_en.en_or import EN_OR
from lexicon_en.en_pa import EN_PA
from lexicon_en.en_pr import EN_PR
from lexicon_en.en_ps import EN_PS
from lexicon_en.en_qas import EN_QAS
from lexicon_en.en_rci import EN_RCI
from lexicon_en.en_re import EN_RE
from lexicon_en.en_svp import EN_SVP

emosent = EmojiSentiment()
sid = SentimentIntensityAnalyzer()
emojiex = Emoji()
swear = Swear()
en_an = EN_AN()
en_asf = EN_ASF()
en_asm = EN_ASM()
en_cds = EN_CDS()
en_ddf = EN_DDF()
en_ddp = EN_DDP()
en_dmc = EN_DMC()
en_is = EN_IS()
en_om = EN_OM()
en_or = EN_OR()
en_pa = EN_PA()
en_ps = EN_PS()
en_pr = EN_PR()
en_qas = EN_QAS()
en_rci = EN_RCI()
en_re = EN_RE()
en_svp = EN_SVP()


class featureManager(object):

    def create_feature_space(self,tweets, train_tweets, featureset=None):
    
    
            global_featureset={
                "BoW"  : self.get_BoW_features(tweets,train_tweets),
                #"hashtagBOW" : self.get_hashtag_features(tweets,train_tweets),
                #"emojiBOW" : self.get_emoji_bow(tweets,train_tweets),
                #"swearBOW" : self.get_swearing_bow(tweets,train_tweets),
                #"posTagBoW" : self.pos_tag_bow(tweets,train_tweets),
                #"adjective" : self.get_adjective_bow(tweets,train_tweets),
                #"numhashtag" : self.get_numhashtag_features(tweets,train_tweets),
                #"isRetweet" : self.isRetweet(tweets, train_tweets),
                #"capitalWordCount" : self.capitalWordCount(tweets, train_tweets),
                #"ironicBOW" : self.get_ironic_bow(tweets, train_tweets),
                #"sexistBOW" : self.get_sexist_bow(tweets, train_tweets),
                #"es_an" : self.get_es_an(tweets, train_tweets),
                #"es_asf" : self.get_es_asf(tweets, train_tweets),
                #"es_asm" : self.get_es_asm(tweets, train_tweets),
                #"es_cds" : self.get_es_cds(tweets, train_tweets),
                #"es_ddf" : self.get_es_ddf(tweets, train_tweets),
                #"es_ddp" : self.get_es_ddp(tweets, train_tweets),
                #"es_dmc" : self.get_es_dmc(tweets, train_tweets),
                #"es_is" : self.get_es_is(tweets, train_tweets),
                #"es_om" : self.get_es_om(tweets, train_tweets),
                #"es_or" : self.get_es_or(tweets, train_tweets),
                #"es_pa" : self.get_es_pa(tweets, train_tweets),
                #"es_pr" : self.get_es_pr(tweets, train_tweets),
                #"es_ps" : self.get_es_ps(tweets, train_tweets),
                #"es_qas" : self.get_es_qas(tweets, train_tweets),
                #"es_rci" : self.get_es_rci(tweets, train_tweets),
                #"es_re" : self.get_es_re(tweets, train_tweets),
                #"es_svp" : self.get_es_svp(tweets, train_tweets),
                #"emojiSentiment" : self.getEmojiSentiment(tweets, train_tweets),
                #"emojiIncongruity" : self.getSentimentDistance(tweets, train_tweets),
                #"emojiCount" : self.getEmojiCount(tweets,train_tweets),
                #"emojiPresence" : self.getEmojiPresence(tweets,train_tweets),
                #"questMark" : self.getQuestionMark(tweets,train_tweets),
                #"hasDotDotDot" : self.hasDotDotDot(tweets,train_tweets),
                #"hasQuote" : self.hasQuote(tweets,train_tweets),
                #"linkCount" : self.getLinkCount(tweets,train_tweets),
                #"textLength" : self.getTextLength(tweets,train_tweets),
                #"emoticonCount" : self.getEmoticonCount(tweets,train_tweets),
                #"ironicWordCount" : self.getIronicWordCount(tweets,train_tweets),
                #"repeatedCharCount" : self.getMentionCount(tweets, train_tweets),
                #"mentionCount" : self.getMentionCount(tweets, train_tweets),
                #"linkPresence" : self.getLinkPresence(tweets,train_tweets),
                #"sentimentScore" : self.sentimentScore(tweets,train_tweets),
                #"verbCount" : self.verbCount(tweets,train_tweets),
                #"nounCount" : self.sentimentScore(tweets,train_tweets),
                #"adjectiveCount" : self.sentimentScore(tweets,train_tweets),
                #"pronounCount" : self.pronounCount(tweets,train_tweets),
                #"conjunctionCount" : self.sentimentScore(tweets,train_tweets),
                #"prepositionCount" : self.sentimentScore(tweets,train_tweets),
                #"adverbCount" : self.sentimentScore(tweets,train_tweets),
                #"resolveURL" : self.resolveURL(tweets,train_tweets),
            }
    
            all_feature_names=[]
            all_X=[]
            all_Y=[]
    
            for key in featureset:
                X, Y = global_featureset[key]
                #all_feature_names=np.concatenate((all_feature_names,feature_names))
                if all_X!=[]:
                    all_X=csr_matrix(hstack((all_X,X)))
                    all_Y=csr_matrix(hstack((all_Y,Y)))
                else:
                    all_X=X
                    all_Y=Y
    
            return all_X,all_Y
    
    
        
    def get_BoW_features(self, tweets, train_tweets):
    
            tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                              analyzer = "word",
                                          stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
    
            feature  = []
            for tweet in tweets:
                #tweet = self.resolveURL(tweet)
                feature.append(tweet)
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                    #tweet = self.resolveURL(tweet)
                    feature_train.append(tweet)
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
               #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
                
               # print(feature_names)
    
                return X
    
    
    def get_numhashtag_features(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            if(len(re.findall("#", tweet)) > 0):
                feature.append(1)
            else :
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                if(len(re.findall("#", tweet)) > 0):
                    feature_train.append(1)
                else :
                    feature_train.append(0)
                    
            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def isRetweet(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            if(len(re.findall("RT @", tweet)) > 0):
                feature.append(1)
            else :
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                if(len(re.findall("RT @", tweet)) > 0):
                    feature_train.append(1)
                else :
                    feature_train.append(0)
                    
            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    
    def capitalWordCount(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            num = 0
            for char in tweet:
                if char.isupper():
                    num+=1
            feature.append(num)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                num = 0
                for char in tweet:
                    if char.isupper():
                        num+=1
                feature_train.append(num)
                    
            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:

            return csr_matrix(np.vstack(feature))
    
    def getEmoticonCount(self, tweets, train_tweets):

        feature  = []
        emoticons = set([":-)", ":(", ":-(", ":')", ":D", ":-D",":-p","D:"])
        for tweet in tweets:
            splittedTweet = set([]) 
            copiedTweet = tweet
            tweetToken = copiedTweet.split(" ")
            for word in tweetToken:
                word = word.lower()
                splittedTweet.add(word)
            feature.append(len(splittedTweet.intersection(emoticons)))

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                splittedTweet = set([]) 
                copiedTweet = tweet
                tweetToken = copiedTweet.split(" ")
                for word in tweetToken:
                    word = word.lower()
                    splittedTweet.add(word)
                feature_train.append(len(splittedTweet.intersection(emoticons)))

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))
    
    def getIronicWordCount(self, tweets, train_tweets):

        feature  = []
        ironicWords = set(["really","love","lovely","wonderful","hilarious","much","like"])
        for tweet in tweets:
            splittedTweet = set([]) 
            copiedTweet = tweet
            tweetToken = copiedTweet.split(" ")
            for word in tweetToken:
                word = word.lower()
                splittedTweet.add(word)
            feature.append(len(splittedTweet.intersection(ironicWords)))

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                splittedTweet = set([]) 
                copiedTweet = tweet
                tweetToken = copiedTweet.split(" ")
                for word in tweetToken:
                    word = word.lower()
                    splittedTweet.add(word)
                feature_train.append(len(splittedTweet.intersection(ironicWords)))

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))
        
    def getLinkCount(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            feature.append(len(re.findall("http", tweet)))

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                feature_train.append(len(re.findall("http", tweet)))

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def getMentionCount(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            mention = len(re.findall("\@", tweet))
            if mention > 0:
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                mention = len(re.findall("\@", tweet))
                if mention > 0:
                    feature_train.append(1)
                else:
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def getQuestionMark(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            count = len(re.findall("\?", tweet))
            if (count > 0):
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                count = len(re.findall("\?", tweet))
                if (count > 0):
                    feature_train.append(1)
                else:
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))
    
    def getLinkPresence(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            count = len(re.findall("http", tweet))
            if (count > 0):
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                count = len(re.findall("http", tweet))
                if (count > 0):
                    feature_train.append(1)
                else:
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def hasQuote(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            count = len(re.findall(r'\"(.+?)\"',tweet))
            if (count > 0):
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                count = len(re.findall(r'\"(.+?)\"',tweet))
                if (count > 0):
                    feature_train.append(1)
                else:
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))
        
    def hasDotDotDot(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            count = len(re.findall("...", tweet))
            if (count > 0):
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                count = len(re.findall("...", tweet))
                if (count > 0):
                    feature_train.append(1)
                else:
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def verbCount(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            score = sum(1 for word, pos in pos_tag(word_tokenize(tweet)) if pos.startswith('V'))
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                score = sum(1 for word, pos in pos_tag(word_tokenize(tweet)) if pos.startswith('V'))
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    
    def pronounCount(self, tweets, train_tweets):

        feature  = []
        for tweet in tweets:
            score = sum(1 for word, pos in pos_tag(word_tokenize(tweet)) if pos.startswith('PRP'))
            if score > 0:
                feature.append(1)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                score = sum(1 for word, pos in pos_tag(word_tokenize(tweet)) if pos.startswith('PRP'))
                if score > 0:
                    feature_train.append(1)
                else :
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    
    def getSentimentDistance(self, tweets, train_tweets):
        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = emosent.get_emoji_sentiment(cleanedTweet)
            ss = sid.polarity_scores(cleanedTweet)
            textsent = float(ss["compound"])
            distance = textsent - score
            if score != 0.0:
                if textsent == 0.0:
                    feature.append(0)
                elif (score < 0 and textsent > 0) or (score > 0 and textsent < 0):
                    feature.append(1)
                #elif (score > 0 and textsent > 0) and (abs(score-textsent)>0.5):
                #    return 1
                #elif (score < 0 and textsent < 0) and (abs(score-textsent)>0.5):
                #    return 1
                else:
                    feature.append(0)
            else:
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = emosent.get_emoji_sentiment(cleanedTweet)
                ss = sid.polarity_scores(cleanedTweet)
                textsent = float(ss["compound"])
                distance = textsent - score
                if score != 0.0:
                    if textsent == 0.0:
                        feature.append(0)
                    elif (score < 0 and textsent > 0) or (score > 0 and textsent < 0):
                        feature.append(1)
                #elif (score > 0 and textsent > 0) and (abs(score-textsent)>0.5):
                #    return 1
                #elif (score < 0 and textsent < 0) and (abs(score-textsent)>0.5):
                #    return 1
                    else:
                        feature.append(0)
                else:
                    feature.append(0)

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))    
    
    def getEmojiSentiment(self, tweets, train_tweets):
        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            cleanedTweet = emoji.demojize(cleanedTweet)
            score = emosent.get_emoji_sentiment(cleanedTweet)
            #print (score)
            if score == 0:
                feature.append(0)
            elif score > 0:
                feature.append(1)
            else :
                feature.append(2)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                cleanedTweet = emoji.demojize(cleanedTweet)
                score = emosent.get_emoji_sentiment(cleanedTweet)
                if score == 0:
                    feature_train.append(0)
                elif score > 0:
                    feature_train.append(1)
                else :
                    feature_train.append(2)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def getEmojiCount(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            cleanedTweet = emoji.demojize(cleanedTweet)
            score = emojiex.getEmojiCount(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                cleanedTweet = emoji.demojize(cleanedTweet)
                score = emojiex.getEmojiCount(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def getTextLength(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            score = len(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                score = len(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def repeatedCharCount(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            repeat = len(re.findall(r'((\w)\2{2,})', cleanedTweet))
            feature.append(repeat)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                repeat = len(re.findall(r'((\w)\2{2,})', cleanedTweet))
                feature_train.append(repeat)
            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def getEmojiPresence(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = emojiex.getEmojiCount(cleanedTweet)
            if score > 0:
                feature.append(1)
            else :
                feature.append(0)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = emojiex.getEmojiCount(cleanedTweet)
                if score > 0:
                    feature_train.append(1)
                else :
                    feature_train.append(0)

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature))

        else:


            return csr_matrix(np.vstack(feature))
    
    def sentimentScore(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            ss = sid.polarity_scores(cleanedTweet)
            if ss["compound"] == 0:
                feature.append(0)
            elif ss["compound"] > 0:
                feature.append(1)
            else :
                feature.append(2)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                #sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(cleanedTweet)
                if ss["compound"] == 0:
                    feature_train.append(0)
                elif ss["compound"] > 0:
                    feature_train.append(1)
                else :
                    feature_train.append(2)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def get_emoji_bow(self, tweets, train_tweets):
            tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              #stop_words="english",
                                              lowercase=True, #true 0.507 false 0.51
                                              binary=True,
                                              max_features=500000)
    
            feature  = []
            for tweet in tweets:
                #a = (' '.join(re.findall(r"#(\w+)", tweet)))
                #print (a)
                cleanedTweet = emoji.demojize(tweet)
                feature.append(emojiex.getEmojiList(cleanedTweet))
    
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                    cleanedTweet = emoji.demojize(tweet)
                    feature_train.append(emojiex.getEmojiList(cleanedTweet))
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
                #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
    
    
                return X
            
    def get_swearing_bow(self, tweets, train_tweets):
            tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              #stop_words="english",
                                              lowercase=True, #true 0.507 false 0.51
                                              binary=True,
                                              max_features=500000)
    
            feature  = []
            for tweet in tweets:
                #a = (' '.join(re.findall(r"#(\w+)", tweet)))
                #print (a)
                feature.append(swear.getSwearList(tweet))
    
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                    feature_train.append(swear.getSwearList(tweet))
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
                #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
    
    
                return X
    
    def get_ironic_bow(self, tweets, train_tweets):
            feature  = []
            ironicWords = set(["donne","donna","ragaza","signora","signore","femmina","femminile","moglie","casalinga","femminista"])
            for tweet in tweets :
                count = 0
                for i in ironicWords:
                    score = len(re.findall(i, tweet))
                    count = count + score

                feature.append(count)
    
    
            if train_tweets!=None:
    
                feature_train  = []
                ironicWords = set(["women","woman","girl","lady","ladi","female","feminine","wife","housewife","feme","femini"])
                for tweet in train_tweets:
                    count = 0
                    for i in ironicWords:
                        score = len(re.findall(i, tweet))
                        count = count + score
                    feature_train.append(count)
                
                return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

            else:


                return csr_matrix(np.vstack(feature))
    
    def get_sexist_bow(self, tweets, train_tweets):
            
            feature  = []
            ironicWords = set(["ugly","foxy","cunt","floozy","doll","pussy","chick","babe","bitch","tramp","whore","slut","showgirl","vagina","rape","rapis","fugly","slag","unfuckable","lesbian","fuckstruggle","hysterical","skank"])
            for tweet in tweets :
                count = 0
                for i in ironicWords:
                    score = len(re.findall(i, tweet))
                    count = count + score

                feature.append(count)
    
    
            if train_tweets!=None:
    
                feature_train  = []
                ironicWords = set(["brutta","ciofeca","strafiga","bona","fica","figa","putta","puttana","bambola","gnocca","gallina","pollastrella","pupa","cagna","vegabondo","troia","sgualdrina","velina","vagina","stupro","stupratore","zoccola","inchiavabile","lesbica","sciacquetta","sciattona","lurida","sveltina","troietta","maiale","isterico","buco di culo","cessa","cesso","brutta"])
                for tweet in train_tweets:
                    count = 0
                    for i in ironicWords:
                        score = len(re.findall(i, tweet))
                        count = count + score
                    feature_train.append(count)
                
                return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

            else:


                return csr_matrix(np.vstack(feature))
            

    def get_adjective_bow(self, tweets, train_tweets):
            tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                              #stop_words="english",
                                              lowercase=False, #true 0.507 false 0.51
                                              binary=True,
                                              max_features=500000)
    
            feature  = []
            for tweet in tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                adj = ""
                token = nltk.word_tokenize(cleanedTweet)
                pos = nltk.pos_tag(token)
                for key in pos:
                    s = key[1]
                    if s[0] == "J":
                        adj = adj +" "+ key[0]
                feature.append(adj)
                #print (adj)
    
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                    adj = ""
                    token = nltk.word_tokenize(cleanedTweet)
                    pos = nltk.pos_tag(token)
                    for key in pos:
                        s = key[1]
                        if s[0] == "J":
                            adj = adj +" "+ key[0]
                    feature_train.append(adj)
                        #print (adj)
    
                tfidfVectorizer = tfidfVectorizer.fit(feature_train)
                X_train = tfidfVectorizer.transform(feature_train)
                X_test = tfidfVectorizer.transform(feature)
                #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
    
    
                return X
            
    def get_hashtag_features(self, tweets, train_tweets):
    
    
            tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              #stop_words="english",
                                              lowercase=True, #true 0.507 false 0.51
                                              binary=True,
                                              max_features=500000)
    
            feature  = []
            for tweet in tweets:
                #a = (' '.join(re.findall(r"#(\w+)", tweet)))
                #print (a)
                feature.append(' '.join(re.findall(r"@(\w+)", tweet)))
    
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                    feature_train.append(' '.join(re.findall(r"@(\w+)", tweet)))
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
                #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
    
    
                return X
            
    def pos_tag_bow(self, tweets, train_tweets):
    
    
            tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              #stop_words="english",
                                              lowercase=True, #true 0.507 false 0.51
                                              binary=True,
                                              max_features=500000)
    
            feature  = []
            for tweet in tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                tagged = pos_tag(word_tokenize(tweet))
                combined_tag = ""
                a = 0
                for i in tagged:
                    combined_tag = combined_tag +" "+str(tagged[a][1])
                    a = a+1
                feature.append(combined_tag)
    
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                   cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                   cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                   cleanedTweet.replace("'","")
                   cleanedTweet.replace('"',"")
                   cleanedTweet.replace('/',"")
                   cleanedTweet.replace("\\","")
                   cleanedTweet.lower()
                   tagged = pos_tag(word_tokenize(tweet))
                   combined_tag = ""
                   a = 0
                   for i in tagged:
                       combined_tag = combined_tag +" "+str(tagged[a][1])
                       a = a+1
                   feature_train.append(combined_tag)
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
                #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
    
    
                return X
            
    def get_es_an(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            score = en_an.get_en_an_count(tweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                score = en_an.get_en_an_count(tweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def get_es_asf(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            score = en_asf.get_en_asf_count(tweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                score = en_asf.get_en_asf_count(tweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def get_es_asm(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_asm.get_en_asm_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_asm.get_en_asm_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def get_es_cds(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_cds.get_en_cds_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_cds.get_en_cds_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_ddf(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_ddf.get_en_ddf_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_ddf.get_en_ddf_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_ddp(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_ddp.get_en_ddp_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_ddp.get_en_ddp_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_dmc(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_dmc.get_en_dmc_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_dmc.get_en_dmc_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_is(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_is.get_en_is_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_is.get_en_is_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_om(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_om.get_en_om_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_om.get_en_om_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_or(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_or.get_en_or_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_or.get_en_or_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_pa(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_pa.get_en_pa_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_pa.get_en_pa_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_pr(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_pr.get_en_pr_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_pr.get_en_pr_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_ps(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_ps.get_en_ps_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_ps.get_en_ps_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def get_es_qas(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_qas.get_en_qas_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_qas.get_en_qas_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
        
    def get_es_rci(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_rci.get_en_rci_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_rci.get_en_rci_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_re(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_re.get_en_re_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_re.get_en_re_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))

    def get_es_svp(self, tweets, train_tweets):

        feature  = []

        for tweet in tweets:
            cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
            cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
            cleanedTweet.replace("'","")
            cleanedTweet.replace('"',"")
            cleanedTweet.replace('/',"")
            cleanedTweet.replace("\\","")
            cleanedTweet.lower()
            score = en_svp.get_en_svp_count(cleanedTweet)
            feature.append(score)

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweet).split())
                cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
                cleanedTweet.replace("'","")
                cleanedTweet.replace('"',"")
                cleanedTweet.replace('/',"")
                cleanedTweet.replace("\\","")
                cleanedTweet.lower()
                score = en_svp.get_en_svp_count(cleanedTweet)
                feature_train.append(score)

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_train))

        else:


            return csr_matrix(np.vstack(feature))
    
    def resolveURL(self, tweets, train_tweets):
            
            tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              analyzer = "word",
                                          stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
    
            feature  = []
            for tweet in tweets:
                #tweet = self.resolveURL(tweet)
                #tweetText = "RT @bnixole: bitch shut the fuck up you're fucking your best friends dad https://t.co/1YR6ydZMgc"
                urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
                #print(urls)
                #flag = 0
                link = ""
                for url in urls:
                    try:
                        link = requests.get(url, verify=False, timeout=10).url
                    except :
                        #print ("connection refused")
                        link = "error"
                if link != "":
                    link = str(link)
                    #print (link)
                    structure = link.split("/")
                    urlcont = ""
                    for st in structure :
                        urlcont = urlcont +" "+st
                    #print(urlcont)
                    feature.append(urlcont)
                    
                else:
                    feature.append("error")
    
            if train_tweets!=None:
    
                feature_train  = []
                for tweet in train_tweets:
                #tweet = self.resolveURL(tweet)
                    #tweetText = "RT @bnixole: bitch shut the fuck up you're fucking your best friends dad https://t.co/1YR6ydZMgc"
                    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
                    #print(urls)
                    #flag = 0
                    for url in urls:
                        link = ""
                        try:
                            link = requests.get(url, verify=False, timeout=10).url
                        except :
                            #print ("connection refused")
                            link = "error"     
                    link = str(link)
                    #print (link)
                    structure = link.split("/")
                    urlcont = ""
                    for st in structure :
                        urlcont = urlcont +" "+st
                    feature_train.append(urlcont)
    
                tfidfVectorizer = tfidfVectorizer.fit(feature)
                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_train)
               #feature_names=tfidfVectorizer.get_feature_names()
    
                return X_train, X_test
    
            else:
                tfidfVectorizer = tfidfVectorizer.fit(feature)
    
                X = tfidfVectorizer.transform(feature)
    
                #feature_names=tfidfVectorizer.get_feature_names()
                
               # print(feature_names)
    
                return X
    
        
def make_feature_manager():
    
    features_manager = featureManager()
    
    return features_manager
    
