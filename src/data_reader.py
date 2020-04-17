from pandas import read_csv


def read_data(data_file='data/traindata.csv'):
    return read_csv(data_file, sep='\t')
