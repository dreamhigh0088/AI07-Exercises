#!/usr/bin/env python
# coding: utf-8

# In[54]:


#1. Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[55]:


#2. Load the csv data
df = pd.read_csv('heart.csv')
df.head()


# In[56]:


#3. Datasets summary and type
df.describe()


# In[57]:


df.info()


# In[58]:


# Distribution of patients in datset
df['target'].value_counts()


# In[59]:


#Split into x and y
x= df.drop('target',axis=1)
y= df['target']


# In[60]:


#Split into train, test sets
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, stratify=y, random_state=711)


# In[61]:


#Define min max scaler, transform
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# In[62]:


#Train and fit with Logistic Regression model 
logmodel=LogisticRegression()
logmodel.fit(x_train_scaled,y_train)
logtrain_acc=logmodel.score(x_train_scaled, y_train)
logtest_acc=logmodel.score(x_test_scaled, y_test)
print('Training data accuracy (Logistic):',logtrain_acc)
print('Testing data accuracy (Logistic):',logtest_acc)


# In[63]:


#Train and fit with Linear Regression model 
linmodel=LinearRegression()
linmodel.fit(x_train_scaled,y_train)
lintrain_acc=linmodel.score(x_train_scaled, y_train)
lintest_acc=linmodel.score(x_test_scaled, y_test)
print('Training data accuracy (Linear):',lintrain_acc)
print('Testing data accuracy (Linear):',lintest_acc)


# In[64]:


#Train and fit with SVM polynomial model 
svm_poly = SVC(kernel='poly', degree=4)
svm_poly.fit(x_train_scaled, y_train)
svmtrain_acc=svm_poly.score(x_train_scaled, y_train)
svmtest_acc=svm_poly.score(x_test_scaled, y_test)
print('Training data accuracy (SVM_polynomial):',svmtrain_acc)
print('Testing data accuracy (SVM_polynomial):', svmtest_acc)


# In[73]:


# #Train and fit with RBF kernel, gamma = 0.2
svm_gamma = SVC(kernel='rbf', gamma=0.2)
svm_gamma.fit(x_train_scaled, y_train)
svmgammatrain_acc=svm_gamma.score(x_train_scaled, y_train)
svmgammatest_acc=svm_gamma.score(x_test_scaled, y_test)
print('Training data accuracy (SVM_gamma):',svmgammatrain_acc)
print('Testing data accuracy (SVM_gamma):', svmgammatest_acc)


# In[75]:


#Evaluate the models accuracy
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Linear Regression','SVM- Polynomial k', 'SVM- RBF k' ], 'Training data Accuracy %': [logtrain_acc*100, lintrain_acc*100, svmtrain_acc*100, svmgammatrain_acc*100],'Testing data Accuracy %': [logtest_acc*100, lintest_acc*100, svmtest_acc*100, svmgammatest_acc*100]})
model_ev


# In[ ]:




