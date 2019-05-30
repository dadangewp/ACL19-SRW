# -*- coding: utf-8 -*-

import codecs
import re
from nltk.stem.porter import PorterStemmer

class EN_OR(object):

    en_or=[]

    def __init__(self):
        self.en_or = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('en_or.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = word.lower()
            word = stemmer.stem(word)
            self.en_or.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_en_or_count(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        for word in words:
            stemmed = stemmer.stem(word)
            if stemmed in self.en_or:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    en_or = EN_OR()
    sentiment=en_or.get_en_or_count("fuck boob pussy")
    print(sentiment)