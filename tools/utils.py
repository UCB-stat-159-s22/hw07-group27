import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

#function to one hot encode a data table
#it accepts a dataframe and a column
#this function takes in a dataframe and one hot encodes a column
def one_hot_encoding(data, column_name):
    #create the one hot encoder 
    oh_enc = OneHotEncoder()
    #fit the column with the encoder
    oh_enc.fit(data[[column_name]])
    #create new columns that have been one hot encoded
    dummies = pd.DataFrame(oh_enc.transform(data[[column_name]]).todense(), columns=oh_enc.get_feature_names_out(), index = data.index)
    #removed original column
    data = data.drop(column_name, axis = 1)

    return data.join(dummies)


#simple label encoder function
#it accepts a data frame and a column
#this function takes in a dataframe and column, and the encoder processes the data
def label_encoder(data, column_name):
    #create the label encoder
    label_encoder = preprocessing.LabelEncoder()
    #fit the column with label encoded values
    data[column_name]= label_encoder.fit_transform(data[column_name])
    
    return data


#create plots
#accepts a data frame and column
#this function outputs a density and probability plot and saves them into a folder
def dist_and_prob_plot(data, column_name):
    fig = plt.figure()
    sns.distplot(data[column_name],fit=norm)
    fig.savefig('figures/' + column_name + 'DensityPlot')
    fig2 = plt.figure()
    stats.probplot(data[column_name], plot=plt)
    fig2.savefig('figures/' + column_name + 'ProbabilityPlot')