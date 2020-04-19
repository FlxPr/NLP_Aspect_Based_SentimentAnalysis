packages =['torch', 'tensorflow', 'transformers', 'pytorch-pretrained-bert', 'pytorch-nlp']

import subprocess

def install(package):
    subprocess.call(["pip", "install", package])

for package in packages: 
    install(package)

import pandas as pd
#from src.data_reader import read_data
#import spacy
#from spacy.lang.en import English


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf

class Classifier:
    
    """The Classifier"""
    
    def __init__(self, trainfile,  max_length=128, batch_size=16):
        self.trainfile = trainfile
        self.max_length = max_length
        self.batch_size = batch_size
        
    ## PREPROCESSING ##
        
    def preprocessing(self, trainfile):
        
        df = pd.read_csv(self.trainfile, sep='\t', header=None, names=['polarity', 'aspect', 'target', 'position', 'sentence'])

        # Create sentence and label lists
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        first_sentences = df.sentence.values
        first_tokens = [tokenizer.tokenize(sentence) for sentence in first_sentences]
        first_input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in first_tokens]

        # Auxiliary sentence
        auxiliary_sentences = ["" + aspect + " - " + target 
                               for aspect,target 
                               in list(zip(df.aspect.values, df.target.values))]
        auxiliary_tokens = [tokenizer.tokenize(sentence) for sentence in auxiliary_sentences]
        auxiliary_input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in auxiliary_tokens]

        # Encode labels
        encoder = LabelEncoder()
        labels = encoder.fit_transform(df.polarity.values)

        # Prepare input ids
        input_ids_prepared = [self.tokenizer.prepare_for_model(
            input_ids_0, 
            input_ids_1,
            max_length=self.max_length,
            truncation_strategy='only_first',
            pad_to_max_length=True, 
            return_token_type_ids=True,
            return_attention_mask=True) for 
                              input_ids_0, input_ids_1 in 
                              list(zip(first_input_ids, auxiliary_input_ids))]

        df_input_ids_prepared = pd.DataFrame(input_ids_prepared)

        input_ids = list(df_input_ids_prepared.input_ids.values)
        token_type_ids = list(df_input_ids_prepared.token_type_ids.values)
        attention_masks = list(df_input_ids_prepared.attention_mask.values)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_masks = torch.tensor(attention_masks)
        token_type_ids = torch.tensor(token_type_ids)

        return input_ids, labels, token_type_ids, attention_masks
    
    ## DATALOADER INPUTS ##

    def dataload(self, input_ids, labels, token_type_ids, attention_masks):
        train_data = TensorDataset(input_ids, attention_masks, token_type_ids,labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        return train_dataloader
    
    #############################################
    
    def train(self, trainfile):
        
        print('Processing File...')
        train_inputs, train_labels, train_token_type_ids, train_attention_masks = self.preprocessing()
        
        print('Creating Dataloaders...')
        
        
        print('Training...')       


    def predict(self, datafile):
        test_df = read_data('data/devdata.csv')
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
