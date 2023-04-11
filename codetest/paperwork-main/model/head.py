import numpy as np
import pandas as pd 
import os
import random
import time
from tqdm import tqdm
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, logging

from data.embedding_init import get_tokenizer,get_embedding

class OriginModel(nn.Module):
    def __init__(self,CONFIG):
        super(OriginModel,self).__init__()
        model_name=CONFIG["model_name"]
        self.config=AutoConfig.from_pretrained(model_name)
        self.config.update({"num_hidden_layers":CONFIG["num_hidden_layers"]})
        self.model=AutoModel.from_pretrained(model_name,config=self.config)
        self.drop=nn.Dropout(p=CONFIG["dropout_rate"])
        self.is_embedding=CONFIG["OUTPUT_EMBEDDING"]
        self.linear=nn.Linear(self.config.to_dict()["hidden_size"],2)
           
        # self.dense = nn.Linear(self.config.to_dict()["hidden_size"], self.config.to_dict()["hidden_size"])
        # self.activation = nn.Tanh()
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self,input_ids,attention_mask,labels=None):
        out=self.model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=False)
        last_hidden_state = out[0]
        cls_embeddings = last_hidden_state[:,0]
#         pooled_output = self.dense(cls_embeddings)
#         pooled_output = self.activation(pooled_output)
        
#         out=self.drop(pooled_output)
        if self.is_embedding:
            return cls_embeddings
            
        out=self.drop(cls_embeddings)
        outputs=self.linear(out)
        preds=outputs.squeeze(1)

        if labels!=None:
            loss=self.loss_fn(preds,labels)
            return loss
        else:
            return preds
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
        
class LR(nn.Module):  # Logsitic Regression
    def __init__(self, CONFIG):
        super().__init__()

        glove_matrix_dir = CONFIG["INPUT_DIR"]
        embedding_matrix = get_embedding(CONFIG["COLUMN_NAME"],glove_matrix_dir)

        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0] + 1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.fc = nn.Linear(embed_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, textid,labels = None):
        embedding_text = self.embedding(textid)
        embedding_text = self.embedding_dropout(embedding_text)
        # take mean of embedding as linear input
        embedding_text = torch.mean(embedding_text,dim=1)
        out1 = self.fc(embedding_text)
        out1.squeeze(1)
        if labels != None:
            loss = self.loss_fn(out1,labels)
            return loss
        else:
            return out1
        
        # return torch.softmax(out, dim = 1)

class RNN(nn.Module):
    def __init__(self,CONFIG):
        super(RNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.hidden_size = 128
        self.CONFIG = CONFIG
        glove_matrix_dir = CONFIG["INPUT_DIR"]
        embedding_matrix = get_embedding(CONFIG["COLUMN_NAME"],glove_matrix_dir)

        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0] + 1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.rnn = nn.RNN(embed_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(4 * self.hidden_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, textid, labels = None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the final_hidden_state of RNN.
        logits.size() = (batch_size, output_size)

        """
        embedding_text = self.embedding(textid)
        embedding_text = self.embedding_dropout(embedding_text)
        embedding_text = embedding_text.permute(1, 0, 2)
        
        batch_size = textid.size()[0]
        h_0 = Variable(torch.zeros(4, batch_size, self.hidden_size).to(self.CONFIG["DEVICE"]))
        output, h_n = self.rnn(embedding_text, h_0)
        # h_n.size() = (4, batch_size, hidden_size)
        h_n = h_n.permute(1, 0, 2)  # h_n.size() = (batch_size, 4, hidden_size)
        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1] * h_n.size()[2])
        # h_n.size() = (batch_size, 4*hidden_size)
        out1 = self.fc(h_n)  # logits.size() = (batch_size, output_size)
        if labels != None:
            loss = self.loss_fn(out1,labels)
            return loss
        else:
            return out1

# class CNN(nn.Module):
#     def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab,
#                  vocab_size, embedding_length, weights, pre_train, embedding_tune):
#         super(CNN, self).__init__()

#         """
#         Arguments
#         ---------
#         batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
#         output_size : 2 = (pos, neg)
#         in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
#         out_channels : Number of output channels after convolution operation performed on the input matrix
#         kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
#         keep_probab : Probability of retaining an activation node during dropout operation
#         vocab_size : Size of the vocabulary containing unique words
#         embedding_length : Embedding dimension of GloVe word embeddings
#         weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
#         --------

#         """
#         self.batch_size = batch_size
#         self.output_size = output_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_heights = kernel_heights
#         self.stride = stride
#         self.padding = padding
#         self.vocab_size = vocab_size
#         self.embedding_length = embedding_length

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
#         if pre_train:
#             self.word_embeddings.weight = nn.Parameter(weights, requires_grad=embedding_tune)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
#         self.dropout = nn.Dropout(keep_probab)
#         self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)

#     def conv_block(self, input, conv_layer):
#         conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
#         activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
#         max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
#         return max_out

#     def forward(self, input_sentences, batch_size=None):
#         """
#         The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
#         whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
#         We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
#         and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
#         to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

#         Parameters
#         ----------
#         input_sentences: input_sentences of shape = (batch_size, num_sequences)
#         batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

#         Returns
#         -------
#         Output of the linear layer containing logits for pos & neg class.
#         logits.size() = (batch_size, output_size)

#         """

#         input = self.word_embeddings(input_sentences)
#         # input.size() = (batch_size, num_seq, embedding_length)
#         input = input.unsqueeze(1)
#         # input.size() = (batch_size, 1, num_seq, embedding_length)
#         max_out1 = self.conv_block(input, self.conv1)
#         max_out2 = self.conv_block(input, self.conv2)
#         max_out3 = self.conv_block(input, self.conv3)

#         all_out = torch.cat((max_out1, max_out2, max_out3), 1)
#         # all_out.size() = (batch_size, num_kernels*out_channels)
#         fc_in = self.dropout(all_out)
#         # fc_in.size()) = (batch_size, num_kernels*out_channels)
#         logits = self.label(fc_in)
#         return logits

class SelfAttention(nn.Module):
    def __init__(self, CONFIG):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """
        self.hidden_size = 128
        self.CONFIG = CONFIG
        glove_matrix_dir = CONFIG["INPUT_DIR"]
        embedding_matrix = get_embedding(CONFIG["COLUMN_NAME"],glove_matrix_dir)

        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0] + 1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.dropout = 0.5
        self.bilstm = nn.LSTM(embed_size, self.hidden_size, dropout=self.dropout, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2 * self.hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * self.hidden_size, 2000)
        self.fc = nn.Linear(2000, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, textid, labels = None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """
        embedding_text = self.embedding(textid)
        embedding_text = self.embedding_dropout(embedding_text)
        embedding_text = embedding_text.permute(1, 0, 2)

        batch_size = textid.size()[0]

        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(self.CONFIG["DEVICE"]))
        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(self.CONFIG["DEVICE"]))

        output, (h_n, c_n) = self.bilstm(embedding_text, (h_0, c_0))
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        out1 = self.fc(fc_out)
        # logits.size() = (batch_size, output_size)

        if labels != None:
            loss = self.loss_fn(out1,labels)
            return loss
        else:
            return out1

class LSTM(nn.Module):
    def __init__(self,CONFIG):
        super(LSTM, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.hidden_size = 128
        self.CONFIG = CONFIG
        glove_matrix_dir = CONFIG["INPUT_DIR"]
        embedding_matrix = get_embedding(CONFIG["COLUMN_NAME"],glove_matrix_dir)
        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0] + 1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm = nn.LSTM(embed_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, textid, labels = None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        embedding_text = self.embedding(textid)
        embedding_text = self.embedding_dropout(embedding_text)
        embedding_text = embedding_text.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        
        batch_size = textid.size()[0]
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.CONFIG["DEVICE"]))
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.CONFIG["DEVICE"]))
        output, (final_hidden_state, final_cell_state) = self.lstm(embedding_text, (h_0, c_0))
        final_output = self.fc(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        if labels != None:
            loss = self.loss_fn(final_output,labels)
            return loss
        else:
            return final_output

def get_model(model_name):
    
    print(f"get model : {model_name}")
    if model_name == "bert-base-uncased":
        return OriginModel
    else:
        return eval(model_name)
    
    
    

if __name__=='__main__':
    pass