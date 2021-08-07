#!/usr/bin/env python
# coding: utf-8

# SIMPLE LINEAR REGRESSION
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[95]:


#Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[102]:


#Importing the dataset
dataset = pd.read_csv("C:\\Users\\SONY\\downloads\\sparkdata.csv")
dataset


# In[54]:


#Plotting the distribution
plt.scatter(dataset.Hours, dataset.Scores, color = 'blue')
plt.title('No. of hours v/s percentage')
plt.xlabel("No. of hours of study")
plt.ylabel("Percentage")


# In[20]:


#Preparing the data
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[21]:


print(X)


# In[22]:


print(Y)


# In[72]:


#Splitting the dataset Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state =0)


# In[73]:


print(X_train)


# In[74]:


print(X_test)


# In[75]:


print(Y_train)


# In[76]:


print(Y_test)


# In[77]:


#Training the SLR Model on Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[78]:


#Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[79]:


#Plotting the Test set results
plt.scatter(Y_test, y_pred)


# In[80]:


#Visualising the Training Set Results
plt.scatter(X_train, Y_train, color ='red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('No. of hours v/s percentage')
plt.xlabel("No. of hours of study")
plt.ylabel("Percentage")


# In[81]:


#Visualising the Test Set Results
plt.scatter(X_test, Y_test, color ='red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('No. of hours v/s percentage')
plt.xlabel("No. of hours of study")
plt.ylabel("Percentage")


# In[82]:


#Evaluating the model
from sklearn import metrics
metrics.mean_absolute_error(Y_test, y_pred)


# In[85]:


#Actual V/S Predicted Data
df=pd.DataFrame({"Actual" : Y_test, 'Predicted' : y_pred})
df


# In[84]:


#Predict the scores when the no. of hours of study is 9.25 hours/day.
Hours = [[9.25]]
predict = regressor.predict(Hours)
print('Score:', predict)


# In[ ]:





# In[ ]:




