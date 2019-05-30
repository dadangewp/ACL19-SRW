# -*- coding: utf-8 -*-

import codecs
import re
from nltk.stem.porter import PorterStemmer

class EN_PR(object):

    en_pr=[]

    def __init__(self):
        self.en_pr = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('en_pr.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = word.lower()
            word = stemmer.stem(word)
            self.en_pr.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_en_pr_count(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        for word in words:
            stemmed = stemmer.stem(word)
            if stemmed in self.en_pr:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    en_pr = EN_PR()
    sentiment=en_pr.get_en_pr_count("whore slut")
    print(sentiment)