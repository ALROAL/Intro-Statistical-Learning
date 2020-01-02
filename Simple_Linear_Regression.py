#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math


# In[2]:


df = pd.read_csv("Advertising.csv", index_col = 0)
df.head()


# In[3]:


plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.scatter(df["Sales"], df["TV"])


# In[4]:


reg = linear_model.LinearRegression()
reg.fit(df[["TV"]], df[["Sales"]])


# In[5]:


plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.scatter(df["TV"], df["Sales"])
plt.plot(df["TV"], reg.predict(df[["TV"]]), color = "red")


# In[126]:


rss = ((df["Sales"].values-reg.predict(df[["TV"]]).reshape(1,-1))**2).sum()
rss


# In[7]:


df["Sales"].mean()


# In[8]:


reg.coef_


# In[9]:


reg.intercept_


# In[10]:


reg.predict(df[["TV"]]).reshape(1,-1)


# In[11]:


df["Sales"].values


# In[111]:


rse = math.sqrt(rss/(len(df)-2))
rse


# In[13]:


tss = ((df["Sales"].values - df["Sales"].mean())**2).sum()


# In[107]:


r2 = 1 - (rss/tss)
r2
se2 = rse**2/(((df["TV"].values - df["TV"].mean())**2).sum())
se = math.sqrt(se2)
se


# In[119]:


from scipy import stats
tstat = reg.coef_/se
tstat
pval = stats.t(len(df)-1).sf(tstat)
pval


# In[120]:


tss = ((df["Sales"].values - df["Sales"].values.mean())**2).sum()


# In[133]:


rss = ((df["Sales"].values-reg.predict(df[["TV"]]).reshape(1,-1))**2).sum()
rss


# In[132]:


fstat = ((tss-rss)/1)/(rss/(len(df)-1-1))
fstat


# In[ ]:




