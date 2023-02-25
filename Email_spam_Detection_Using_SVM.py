#!/usr/bin/env python
# coding: utf-8

# Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Data Preprocessing
# 

# In[2]:


#load the email dataset
data = pd.DataFrame(pd.read_csv('spam.csv',encoding="ISO-8859-1"))


# In[3]:


#replacing null values with null string
data = data.where((pd.notnull(data)),'')


# Data Inspection

# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


#labelling sapm as 0 and non-spam as 1
data.loc[data['v1']=='spam','v1',] = 0
data.loc[data['v1']=='ham','v1',] = 1


# In[7]:


data.head()


# separating  columns as x--> v2 and y--> v1

# In[8]:


x = data['v2']
y = data['v1']
print(x)
print("-----------------------------------------------------------")
print(y)


# Splitting data into train and test datasets

# In[9]:


x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=4)


# Feature Extraction

# In[10]:


#tranform the text data to feature vectors using TfidfVectorizer as input to SVM model
#convert all text data to lowercase
Tfidf = TfidfVectorizer(min_df=1,lowercase='True',stop_words='english')
x_train_feaatures = Tfidf.fit_transform(x_train)
x_test_features = Tfidf.transform(x_test)


# In[11]:


#convert y_train and y_test values into int type
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# Building the model 
# 
# Training Model ---> Support Vector Machine

# In[12]:


#training the model with train datasets
model = LinearSVC()
model.fit(x_train_feaatures,y_train)


# Evaluating Model

# In[13]:


#Evaluating the model
#prediction on training data
from sklearn.metrics import accuracy_score
predict_x_train_data = model.predict(x_train_feaatures)
accuracy_on_training_data = accuracy_score(predict_x_train_data,y_train)
#printing accuracy of training data
print(accuracy_on_training_data)


# In[14]:


#prediction on testing data
prediction_on_testing_data = model.predict(x_test_features)
accuracy_on_testing_data = accuracy_score(prediction_on_testing_data,y_test)


# In[15]:


#printing the presiction accuracy of testing dataset
print("Accuracy of testing data",accuracy_on_testing_data)


# Prediction on new email

# In[16]:


#new spam mail
input_mail=['Are you ready to jumpstart your career in Business Analytics? Do you want to learn about the role, skills, and roadmap needed to succeed in this data-driven industry? If yes, we have exciting news for you!.We are proud to announce that we are hosting a webinar on "How to start your career in Business Analytics" on 4th Feb 2023 at 5 PM IST, where our expert speaker, Diksha Mohnani, will share her insights and guide you through the essential aspects of this field.']
#convert text to vector
input_mail_features = Tfidf.transform(input_mail)
#prediction on new mail
predict = model.predict(input_mail_features)
if predict==1:
  print("HAM MAIL")
else:
  print("SPAM MAIL")

