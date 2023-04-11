import os 
import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import nltk
import json
import pickle

import torch
from torchtext.data.utils import ngrams_iterator
from keras.preprocessing import text, sequence
from keras_preprocessing.sequence import pad_sequences

from data.read_data import read_data_file

def get_all_text(head,input_dir):

    # read positive data
    positive_train_file = os.path.join(input_dir,"PositiveData")
    
    positive_train_df = read_data_file(positive_train_file,False,5000,head = head)
    positive_train_df["label"] = positive_train_df["label"].astype(int)

    # read negative data
    negative_train_file = os.path.join(input_dir,"NegativeData")

    negative_train_df = read_data_file(negative_train_file,False,5000,head = head)
    negative_train_df["label"] = negative_train_df["label"].astype(int)

    train_df = pd.concat([positive_train_df,negative_train_df],axis = 0)

    keyword_index = head.index("keyword")
    train_df.apply(lambda row: row[0].replace("[MASK]",row[keyword_index]),axis = 1)
    all_text = train_df["Query"]
    return all_text

def build_vocab(xlist, NGRAMS, min_count):
    vocabi2w = ['[SOS]', '[EOS]', '[PAD]', '[UNK]']  # A list of unique words
    seen = defaultdict(int)
    for i in tqdm(range(len(xlist)),ncols = 20):
        tokens = nltk.word_tokenize(xlist[i][0])
        tokens = tokens if NGRAMS  ==  1 else ngrams_iterator(tokens, NGRAMS)
        for token in tokens:
            seen[token] += 1
    vocabi2w += [x for x in seen if seen[x] >=  min_count]
    vocabw2i = {vocabi2w[x]:x for x in range(len(vocabi2w))}
    return vocabw2i, vocabi2w
    
def get_vocab(all_text,CONFIG,NGRAMS = 1):
    
    vocab_path = CONFIG["INPUT_DIR"]
    if not os.path.exists(os.path.join(vocab_path,"vocab_dict.json")):
        os.mkdir(vocab_path)
        unique_vocab_dict, unique_vocab_list = build_vocab(all_text, NGRAMS, min_count = 10)
        with open(os.path.join(vocab_path,"vocab_dict.json"),"w",encoding = "UTF-8") as tf:
            json.dump(unique_vocab_dict,tf)

        with open(os.path.join(vocab_path,"vocab_list.json"),"w",encoding = "UTF-8") as tf:
            json.dump(unique_vocab_list,tf)
    else:
        with open(os.path.join(vocab_path,"vocab_dict.json"),"r",encoding = "UTF-8") as tf:
            unique_vocab_dict = json.load(tf)

        with open(os.path.join(vocab_path,"vocab_list.json"),"r",encoding = "UTF-8") as tf:
            unique_vocab_list = json.load(tf)
    
    return unique_vocab_dict, unique_vocab_list

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype = 'float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def get_tokenizer(head = None,tokenizer_dir = None):
    tokenizer_path=os.path.join(tokenizer_dir,"tokenizer.pickle")

    if not os.path.exists(tokenizer_path):
        input_dir = tokenizer_dir
        all_text = get_all_text(head,input_dir)
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(all_text)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'tokenizer are generated ')
        
    else:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
    return tokenizer

def get_embedding(head,glove_matrix_dir):

    tokenizer_dir = glove_matrix_dir
    tokenizer = get_tokenizer(head,tokenizer_dir)
    max_features = None
    max_features = max_features or len(tokenizer.word_index) + 1

    embedding_dir = "/root/autodl-tmp/glove"
    embedding_path = os.path.join(embedding_dir,"glove.6B.100d.txt")
    glove_matrix_path = os.path.join(glove_matrix_dir,"glove_matrix")
    
    if os.path.exists(glove_matrix_path):
        glove_matrix = torch.load(glove_matrix_path)
    else:
        glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, embedding_path)
        torch.save(glove_matrix,glove_matrix_path)
        print(f'glove matrix are generated ')
        print('n unknown words (glove): ', len(unknown_words_glove))

    return glove_matrix
    