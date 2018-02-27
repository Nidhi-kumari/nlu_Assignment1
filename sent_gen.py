import nltk
import math
from sklearn.model_selection import train_test_split
from nltk.corpus import brown
from nltk.corpus import gutenberg

txt_files_g=nltk.corpus.gutenberg.fileids()
corpus_g = gutenberg.sents(txt_files_g)
txt_files_b = nltk.corpus.brown.fileids()
corpus_b = brown.sents(txt_files_b)

train_g,test=train_test_split(corpus_g, test_size=0.2)
train_bg= train_g + corpus_b

def remove_punctuation(corpus):
    cleaned_corpus=[]
    
    punctuations = ['!','(',')','-','[',']','{',';',':',"'",'\\','<','>','.','/','?','~','&',"''",',','--','``','"']
    for sent in corpus :
        sent1=[]
        for word in sent:
            if word not in punctuations:
                sent1.append(word)
        cleaned_corpus.append(sent1)  
    return cleaned_corpus


cleaned_train=remove_punctuation(train_bg)


import nltk
import math
Start='*'
Stop='STOP'

unigram_p = {}
bigram_p = {}
trigram_p = {}
def calc_probabilities(brown):
    
   
    sentCount = len(brown);
    for sent in brown:
        tokens = sent[:-1]         
        tokens = [Start] + [Start] + tokens + [Stop]
       
        for i, tok in enumerate(tokens):
            if(i >= 2):
                uni_tuple = tuple([tok])
                if(uni_tuple in unigram_p):
                    unigram_p[uni_tuple] = unigram_p[uni_tuple]+1
                else:
                    unigram_p[uni_tuple] = 1
            
            if(i >= 1 and i < len(tokens)-1):
                bi_tuple = tuple([tok, tokens[i+1]])
                if(bi_tuple in bigram_p):
                    bigram_p[bi_tuple] = bigram_p[bi_tuple]+1
                else:
                    bigram_p[bi_tuple] = 1

            if(i < len(tokens)-2):
                tri_tuple = tuple([tok, tokens[i+1],tokens[i+2]])
                if(tri_tuple in trigram_p):
                    trigram_p[tri_tuple] = trigram_p[tri_tuple]+1
                else:
                    trigram_p[tri_tuple] = 1
   
    for tri in trigram_p:
        if(tri[0]=="*" and tri[1]=="*"):
            base = sentCount
        else:
            bi_tuple = tuple([tri[0],tri[1]])
            base = bigram_p[bi_tuple]
        prob = 1.0*trigram_p[tri] / base
        #prob = math.log(prob,2)
        trigram_p[tri] = prob

    
calc_probabilities(train_bg)
print(trigram_p)

   


       
highest_order=3
def get_context(sentence):
       
        return sentence[(len(sentence) - highest_order + 1):]
    
import numpy as np
def generate_next_word(sent, trigram_p):
        context = tuple(get_context(sent))
        pos_ngrams = list((ngram, np.log2(trigram_p.get(ngram))) for ngram in trigram_p
            if ngram[:-1] == context)
        
        _, max_logprob = max(pos_ngrams, key=lambda x: x[1])
        pos_ngrams = list(
            (ngram, math.exp(prob - max_logprob)) for ngram, prob in pos_ngrams)
        total_prob = sum(prob for ngram, prob in pos_ngrams)
        pos_ngrams = list(
            (ngram, prob/total_prob) for ngram, prob in pos_ngrams)
        rand = random.random()
        for ngram, prob in pos_ngrams:
            rand -= prob
            if rand < 0:
                return ngram[-1]
        return ngram[-1]
    
    
import random
def generate_sentence( min_length=10):
       
        sent = []
        probs = trigram_p
        while len(sent) < min_length + highest_order:
            sent = ['*'] * (highest_order - 1)
            
            sent.append(generate_next_word(sent, probs))
            while sent[-1] != 'STOP':
                sent.append(generate_next_word(sent, probs))
        sent = ' '.join(sent[(highest_order - 1):-1])
        return sent

sent=generate_sentence()
print(sent)


