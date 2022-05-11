#!/usr/bin/env python
# coding: utf-8

# # What Do Home-Buyer Care About? 

# In[1]:


import pandas as pd


# What do home-buyer care about? This is a question that any home-seller will be interested! The answer for this question will make sure that home-seller puts much more influence in price-negotiation with buyers! We will try to find the answer for this question by exploring dataset about House Price, which has 79 explanatory variables describing every aspect of residential homes in Ames, Iowa. We will mainly use **Random Forest** model to figure out which variables are the most important one to decide the housing price in Ames.

# # Exploratory Data Analysis on the House_Price.csv

# In[2]:


data.head()


# In[6]:


data.shape


# The House_Price.csv is consist of 81 columns and 1460 rows. However, we will only use 79 columns for our feature because it includes ID columns, whcih is useless and SalePrice is our output columns, which should not be included as one of the features. First of all, we will use label and one-hot encoding to convert our categorical varibles into dummy variables so that we can start the analysis. We mixed both encoding method, since we figured out that for ordinal categorical variable, the label encoding works better, and for cardinal categorical variable, one-hot encoding works better.
# 
# ### Feature Correlation 
# 
# First, we will observe the dataset, and see if we have any inter-relationship between feature that shows high correlation. We don't have to do any feature selection because first, we don't have lots of features to investigate, and second, the random forest model's accuracy doesn't get suffered from collinearity. Despite of these fact, we want to observe its inter-correlation since observing highly interacting variables might give us some hint to our question, and sometimes removing highly correlated variables help us to mitigate possible danger of overfitting. 

# This is the correlation matrix and list of features that have high inter-correlation between the other variables (more than 0.5).  

# ![correlation_plot](figures/combined_data_heatmap.png)

# In[10]:


high_cor = pd.read_csv('data/top_cor.csv')
high_cor


# This result conveys two important information regarding this dataset.
# 
# 1. We initially expected that many varibales will be highly inter-correlated each other (For example, if overall quality is high, then external quality and bathroom quality will be high as well), but suprisingly we have very few of them are correlated. Lots of variable have inter-correlation less than 0.5.
# 
# 2. It is interesting to see that some qualities are negatively correlated with other variables, while overall qualities shows the highest correlation with other variables. Only variables on quality shows some unexpected behavior than other variables.
# 
# From our correlation matrix, we can conclude that we are good to use this dataset for our analysis, since we don't have too many correlating variables. 
# Also, it looks like we don't have to eliminate any feature, since every variables has correlation less than 0.8.

# ### Observation on SalePrice

# We also want to observe whether our SalePrice is normally distributed or not. In many occasion, normality of output variable doe not matter in building random forest model, but since we are more interested in finding variable importance than building a predictive model, it will be much safer if we normalize our output variables. 

# ![density_plot](figures/SalePriceDensityPlot.png)

# ![qq_plot](figures/SalePriceProbabilityPlot.png)

# From the density plot and and Q-Q plot (Probability Plot), we can observe that we have two issues in normality. 
# 
# 1. Our graph is right-skewed that it might need some transformation to center this more.
# 2. According to our probability plot, our graph is very not normal in right ends of the distribution.
# 
# In order to resolve this issue, log-transformation should be done!

# ![density_plot](figures/Log_SalePriceDensityPlot.png)

# ![qq_plot](figures/Log_SalePriceProbabilityPlot.png)

# After our log transformation, we can observe that all problems that we talked about previously solved perfectly. Our graph now looks very normal with good center. We will use our transfomed ourput to fit our random forest model.

# # Fitting a Random Forest Model and Variable Improtance 

# In order to fit our random forest model, we are seperating our dataset into training and test dataset, so that later, we can calculate MSE to validate our model and its result regarding variable importance.

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
#     rf = RandomForestRegressor(n_estimators=500)
#     rf.fit(X_train, y_train)

# After we fit our random forest model using the training set, we can observe which variables brought the biggest decrease in MSE when our random forest model tried to predict the result. We found top 45 variables that brought the large decrease in MSE, which means they are the most important variables in determining the sale price of the house.

# ![45_var](figures/Top_45_Combined_Encoding_Feature_Importance.png)

# From this chart, we can observe that some variables are definitely more important than other variable in determining sale price of the house. However, we can also see that some variable shows very small decrease in MSE and they don't even need to be included in our chart. Therefore, we will adjust our threshold level from 0.001 to 0.01 so that we can only observe 11 important variables. 

# In[19]:


top_11_var = pd.read_csv('data/top_11_variable.csv')
top_11_var


# ![45_var](figures/Top_11_Combined_Encoding_Feature_Importance.png)

# From this chart, we can observe that the top 5 most important variables in determining Sales Price of House are 
# 
# 1. OverallQual - Overall material and finish quality
# 2. GrLivArea - Above grade (ground) living area square feet
# 3. BsmFinSF1 - Type 1 finished square feet
# 4. 1stFlrSF - First Floor square feet
# 5. TotalBsmtSF - Total square feet of basement area
# 
# This is somewhat very obvious result that two most important quality in determining house price are quality and size of the house. It shows that living room, first floor and basement are the three most important rooms of the house in determining the house price. Interestingly, year bulit and whether the house has air condition or not was very important features in determing the house price.

# In[22]:


error = pd.read_csv('data/Combined_Error_Table.csv')
error


# If we examine error of our model, we can see that our model was pretty good model in predicting the house price, so our variable importance plot will be most likely to be accurate as well. 

# ## Author Contributions:
# 
# Isaac: Initial Code with Random Forest Analysis. Writing scripts for main.ipynb
# 
# Brandon:
# 
# Raj: 

# In[ ]:





# In[ ]:





# In[ ]:




