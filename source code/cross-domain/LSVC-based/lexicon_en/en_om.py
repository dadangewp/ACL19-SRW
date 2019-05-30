# -*- coding: utf-8 -*-


import codecs
import re
from nltk.stem.porter import PorterStemmer

class EN_OM(object):

    en_om=[]

    def __init__(self):
        self.en_om = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('en_om.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = word.lower()
            word = stemmer.stem(word)
            self.en_om.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_en_om_count(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        for word in words:
            stemmed = stemmer.stem(word)
            if stemmed in self.en_om:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    en_om = EN_OM()
    sentiment=en_om.get_en_om_count("fuck boob pussy")
    print(sentiment)