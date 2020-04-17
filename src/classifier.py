import pandas as pd
from src.data_reader import read_data
from spacy.lang.en import English
import spacy



class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        train_df = read_data('data/traindata.csv')
        """Trains the classifier model on the training set stored in file trainfile"""


    def predict(self, datafile):
        test_df = read_data('data/devdata.csv')
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """





