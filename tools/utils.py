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

#function to one hot encode
def one_hot_encoding(data, column_name):

    oh_enc = OneHotEncoder()
    oh_enc.fit(data[[column_name]])
    dummies = pd.DataFrame(oh_enc.transform(data[[column_name]]).todense(), columns=oh_enc.get_feature_names_out(), index = data.index)
    data = data.drop(column_name, axis = 1)

    return data.join(dummies)


#simple label encoder function
def label_encoder(data, column_name):
    
    label_encoder = preprocessing.LabelEncoder()
    data[column_name]= label_encoder.fit_transform(data[column_name])
    
    return data


#create plots
def dist_and_prob_plot(data, column_name):
    fig = plt.figure()
    sns.distplot(data[column_name],fit=norm)
    fig.savefig('figures/' + column_name + 'Density Plot')
    fig2 = plt.figure()
    stats.probplot(data[column_name], plot=plt)
    fig2.savefig('figures/' + column_name + 'Probability Plot')