import nltk
import math
from sklearn.model_selection import train_test_split

from nltk.corpus import brown
txt_files = nltk.corpus.brown.fileids()
corpus = brown.sents(txt_files)

# breaking the corpus in train,test 
corpus_size=len(corpus)
train1,test=train_test_split(corpus, test_size=0.2)


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

cleaned_train=remove_punctuation(train1)
train,validation=train_test_split(list(cleaned_train), test_size=0.2)

vocab=[]
for i in train:
    for j in i:
        if j not in vocab:
            vocab.append(j)

vocab_set=set(vocab)
cleaned_test=remove_punctuation(test)

import nltk
import math
from collections import Counter
Start='*'
Stop='STOP'

unigram_c={}
bigram_c={}
trigram_c={}
unigram_p = {}
bigram_p = {}
trigram_p = {}

UNK="UNK"
unks = set()
def calc_probabilities(corpus):
    
   
    sentCount = len(corpus);
    for sent in corpus:
        tokens = sent[:-1]         
        tokens = [Start] + [Start] + tokens + [Stop]
        max_UNK=0
        for i, tok in enumerate(tokens):
            if(i >= 2):
                uni_tuple = tuple([tok])
                if(uni_tuple in unigram_c):
                    unigram_c[uni_tuple] = unigram_c[uni_tuple]+1
                else:
                    unigram_c[uni_tuple] = 1
                    
                    
    for unigram, count in unigram_c.items():
        if (count == 1 and max_UNK<=1000) :
            unks.add(unigram[0])
            max_UNK+=1
            
    for word in unks:
        del unigram_c[(word,)]

    unigram_c[(UNK,)] = max_UNK
            
            
    for sent in corpus:  
                     
            
            tokens0 = [token if token not in unks else UNK for token in sent[:-1]]
            tokens = [Start] + [Start] + tokens0 + [Stop]
            for i, tok in enumerate(tokens):
                if(i >= 1 and i < len(tokens)-1):
                    bi_tuple = tuple([tok, tokens[i+1]])
                    #add_n_gram_counts(bi_tuple, nex,pre)
                    if(bi_tuple in bigram_c):
                        bigram_c[bi_tuple] = bigram_c[bi_tuple]+1
                    else:
                        bigram_c[bi_tuple] = 1

                if(i < len(tokens)-2):
                    tri_tuple = tuple([tok, tokens[i+1],tokens[i+2]])
                    if(tri_tuple in trigram_c):
                        trigram_c[tri_tuple] = trigram_c[tri_tuple]+1
                    else:
                        trigram_c[tri_tuple] = 1

    
    for tri in trigram_c:
        if(tri[0]=="*" and tri[1]=="*"):
            base = sentCount
        else:
            bi_tuple = tuple([tri[0],tri[1]])
            base = bigram_c[bi_tuple]
        prob = 1.0*trigram_c[tri] / base
        #prob = math.log(prob,2)
        trigram_p[tri] = prob
    
    for bi in bigram_c:
        if(bi[0]=="*"):
            base = sentCount
        else:
            uni_tuple = tuple([bi[0]])
            base = unigram_c[uni_tuple]
        prob = 1.0*bigram_c[bi] / base
        #prob = math.log(prob,2)
        bigram_p[bi] = prob
        
    total = 0   
    for uni in unigram_c:
        total = total + unigram_c[uni]
    for uni in unigram_c:
        prob = 1.0*unigram_c[uni] / total
        #prob = math.log(prob,2)
        unigram_p[uni] = prob

   


calc_probabilities(train)

def linear_interpolation1(l1,l2,l3,test_data):
   
    prob=0
    total=0
    
    #l1=1-(l2+l3)
    L1=0.95 #give 0.95 weightage to word if it is in vocab
    L2=.05  #give 0.05 weightage to word if it is in vocab
    
    
   
    
    
    for sentences in test_data:
    
        prob_sent=1
        tokens = [token if (token,) in unigram_p else UNK for token in sentences[:-1]]
        #tokens = sentences[:-1]         
        words = [Start]  +[Start]+ tokens + [Stop]
        
        total+=len(tokens)
        
        for index,token in enumerate(words):
            p=0
            if(index<2):
                continue
          
            
            bigram=tuple([words[index-1],token])
            unigram=tuple([token])
            trigram=tuple([words[index-2],words[index-1],token])
            #unigram_p[Start] = len(test_data)
            #bigram_p[('*', '*')] = len(test_data)
          
        
            tri_p=0
            if(trigram in trigram_p):
                tri_p=trigram_p[trigram]
            bi_p=0  
            if(bigram in bigram_p):   
                bi_p=bigram_p[bigram]
            uni_p=0   
            if(unigram in unigram_p):
                uni_p=unigram_p[unigram]
                
            
            
            
            
            p=l1*uni_p+l2*bi_p+l3*tri_p
            
            
                
            prob_sent=prob_sent*p
            
            
        if(prob_sent>0):
            prob+=math.log(prob_sent,2)
                
               
    prob /= total
    perplexity = 2 ** (-1 * prob)
    #print(l1,l2,l3)
    #print(perplexity)
    return perplexity


l1=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0]
l2=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0]
l3=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0]
plx={}
for k in l3:
    for j in l2:
        for i in l1:
            if i + j + k == 1:
                #print(k,j,i)
                plx1=linear_interpolation1(i,j,k,validation)
                plx[(i,j,k)]=plx1
                #print(plx)

i,j,k=min(plx, key=plx.get)
print (i,j,k)
plx[min(plx, key=plx.get)]

calc_probabilities(train1)
print("perplexity on test data {}" .format(linear_interpolation1(i,j,k,test)))

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
