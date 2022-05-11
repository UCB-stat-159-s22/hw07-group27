import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tools as tool
from tools import utils as util
from os.path import exists
from os import remove


def test_one_hot_encoder():
    data = pd.read_csv('data/House_Price.csv')
    data = util.one_hot_encoding(data, 'MSZoning')
    assert data.shape == (1460, 85)

def test_label_encoder():
    data = pd.read_csv('data/House_Price.csv')
    data = util.label_encoder(data, 'MSZoning')
    assert data.shape == (1460, 81)

def test_dist_and_prob_plot():
    data = pd.read_csv('data/House_Price.csv')
    util.dist_and_prob_plot(data, 'SalePrice')
    assert exists('figures/' + 'SalePrice' + 'DensityPlot' + '.png')
    assert exists('figures/' + 'SalePrice' + 'ProbabilityPlot' + '.png')
    remove('figures/' + 'SalePrice' + 'DensityPlot' + '.png')
    remove('figures/' + 'SalePrice' + 'ProbabilityPlot' + '.png')