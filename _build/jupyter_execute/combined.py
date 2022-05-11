#!/usr/bin/env python
# coding: utf-8

# ## Data Import 

# In[1]:


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


# In[2]:


data = pd.read_csv('data/House_Price.csv')


# In[3]:


#selecting all columns that do not have numeric values
object_column = data.select_dtypes(include = ['object']).columns.tolist()


# ## Feature Processing (One Hot and Label Encoding)

# In[4]:


data = pd.read_csv('data/House_Price.csv')


# In[5]:


#selecting which columns to use label encoding on
#columns that compare the quality should be used or show that one category is better than
#another category should be used here
lccolumn = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']


# In[6]:


#selecting which columns to use one hot encoding on
#ex: Streets should use the one hot encoder because the type of road 
#access should be based on whether it exists or not. 
ohccolumn = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
             'Exterior1st','Exterior2nd', 'MasVnrType', 'Foundation','Heating','Electrical','GarageType','MiscFeature','SaleType','SaleCondition']


# In[7]:


for column in ohccolumn:
    data = util.one_hot_encoding(data, column)


# In[8]:


for column in lccolumn:
    data = util.label_encoder(data, column)


# In[9]:


combined_data = data


# In[10]:


sns_plot = sns.heatmap(combined_data.corr())
fig = sns_plot.get_figure()
fig.savefig("figures/combined_data_heatmap", bbox_inches = "tight")


# In[11]:


threshold = 0.7

df_corr = combined_data.corr()

columns = np.full((df_corr.shape[0],), True, dtype=bool)

for i in range(df_corr.shape[0]):
    for j in range(i+1, df_corr.shape[0]):
        if df_corr.iloc[i,j] >= threshold:
            if columns[j]:
                columns[j] = False


selected_columns = combined_data.columns[~columns]
high_correlation = combined_data[selected_columns]


# In[12]:


absolute_corr = np.abs(df_corr["SalePrice"])
top_cor = df_corr["SalePrice"][absolute_corr > 0.5]
df_top_cor = top_cor.to_frame()
df_top_cor.to_csv("data/top_cor.csv")


# In[13]:


selected_columns


# In[14]:


high_correlation.corr().head()


# ## Log Sale Price

# In[15]:


util.dist_and_prob_plot(data, 'SalePrice')


# In[16]:


data['Log_SalePrice'] = np.log(data['SalePrice'])
util.dist_and_prob_plot(data, 'Log_SalePrice')


# In[17]:


combined_data['SalePrice'] = np.log(combined_data['SalePrice'])
combined_data = combined_data.drop('Log_SalePrice', axis = 1)


# ## Random Forest Model(One Hot and Label Encoding)

# In[18]:


combined_data = combined_data.dropna()
X = combined_data.drop(['SalePrice', 'Id'], axis = 'columns')
y = combined_data['SalePrice']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)


# In[20]:


#made a dataframe with values greater than .001 importance
#sorted the values to look better on bargraph
dataframe = pd.DataFrame(rf.feature_importances_, rf.feature_names_in_)
a = dataframe[dataframe[0] > .001]
a = a.rename(columns = {0: 'values'})
b = a.sort_values('values', ascending = True)
b.head()


# In[21]:


plt.figure(figsize=(7, 20))
plt.barh(b.index, b['values'])
plt.title('Feature Importance Above .001')
plt.ylabel('Features')
plt.xlabel('Importance')
plt.savefig('figures/Top_45_Combined_Encoding_Feature_Importance', bbox_inches = "tight")


# In[22]:


#made another dataframe with a smaller importance threshold
dataframe = pd.DataFrame(rf.feature_importances_, rf.feature_names_in_)
c = dataframe[dataframe[0] > .01]
c = c.rename(columns = {0: 'values'})
d = c.sort_values('values', ascending = True)
d.to_csv("data/top_11_variable.csv")


# In[23]:


plt.figure(figsize=(7, 20))
plt.barh(d.index, d['values'])
plt.title('Feature Importance Above .01')
plt.ylabel('Features')
plt.xlabel('Importance')
plt.savefig('figures/Top_11_Combined_Encoding_Feature_Importance', bbox_inches = "tight")


# In[24]:


y_output = rf.predict(X_test)


# In[25]:


print('MSE', mean_squared_error(y_test, y_output))
print('RMSE', (mean_squared_error(y_test, y_output))** (1/2))
print('Adj R^2 value:', r2_score(y_test, y_output))


# In[26]:


pd.DataFrame({'Type of Error':['MSE', 'RMSE', 'Adj R^2 Value'],'Value' : [mean_squared_error(y_test, y_output), (mean_squared_error(y_test, y_output))** (1/2),  r2_score(y_test, y_output)]}).set_index('Type of Error').to_csv('data/Combined_Error_Table.csv')

