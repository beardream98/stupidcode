import numpy as np
from math import log2
corpus = ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document']

corpus=[sentence.split() for sentence in corpus]

def build_matrix(corpus):
    m=len(corpus)
    unique_word=set([word for sentence in corpus for word in sentence])
    n=len(unique_word)
    corpus_matrix=np.zeros((m,n))

    for j,word in enumerate(unique_word):
        for i,sentence in enumerate(corpus):
            corpus_matrix[i][j]=sentence.count(word)
    
    return corpus_matrix,n
corpus_matrix,n=build_matrix(corpus)

def tf_idf(i,corpus,corpus_matrix):
    sentence=corpus[i]
    vec=np.zeros(n)
    tf_vec=corpus_matrix[i]/len(sentence)
    doc_num=corpus_matrix.shape[0]
    idf_vec=np.log( (doc_num+1) / (np.count_nonzero(corpus_matrix,0))) +1

    vec=tf_vec*idf_vec
    return vec
vec=tf_idf(0,corpus=corpus,corpus_matrix=corpus_matrix)


            
        