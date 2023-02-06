#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[12]:


x=[4,5,10,4,10,8,3,2,7,3]
y=[23,27,22,30,21,23,27,29,30,25]

data=list(zip(x,y))
data


# In[13]:


model = KMeans(n_clusters =2)
model.fit(data)


# In[16]:


plt.scatter(x,y,c=model.labels_)
plt.show()


# In[21]:


data= pd.read_csv("C:/Users/DITU.DESKTOP-G648QP4/Downloads/datasets-master/titanic.csv")
data


# In[26]:


x=data["Survived"]
y=data["Pclass"]
data=list(zip(x.values,y.values))
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
model.fit(data)


# In[27]:


plt.scatter(x,y,c=model.labels_)
plt.show

