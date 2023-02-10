#!/usr/bin/env python
# coding: utf-8

# In[53]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split


# Data Preprocessing

# In[2]:


#Reading dataset
data = pd.read_csv("CarPrice_Assignment.csv")


# In[3]:


#data table
data.head()


# In[4]:


#checking number of null values in dataset
data.isnull().sum()


# In[5]:


data.info()


# In[10]:


#stastical calculations of dataset
print(data.describe())


# In[13]:


#To know different car companies names in dataset
print(data['CarName'].unique())


# Data Expploration

# In[16]:


#Distribution of Price values
plt.figure(figsize=(10,15))
sn.displot(data['price'])
plt.show()


# In[19]:


#Correlation among the features
correlations = data.corr()
print(correlations)


# In[25]:


plt.figure(figsize=(20, 15))
correlations = data.corr()
sn.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# Model Building

# In[60]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


# In[61]:


x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[55]:


model = LinearRegression()
model.fit(xtrain, ytrain)


# In[56]:


predictions = model.predict(xtest)


# In[57]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




