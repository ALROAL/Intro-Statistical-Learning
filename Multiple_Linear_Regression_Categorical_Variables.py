#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


df = pd.read_csv("Credit.csv", index_col = 0)
df.head()


# In[3]:


df2 = pd.get_dummies(df, columns = ["Gender", "Student","Married","Ethnicity"], drop_first = True)
df2.head(10)


# In[4]:


reg = linear_model.LinearRegression()
reg.fit(df2[["Gender_Female"]],df2[["Balance"]])


# In[5]:


ssr = ((df2["Balance"].values-reg.predict(df2[["Gender_Female"]]).reshape(1,-1))**2).sum()
rse = np.sqrt(ssr/(len(df2)-2))
se = rse/(np.sqrt((((df2["Gender_Female"].values - df2["Gender_Female"].mean())**2).sum())))


# In[6]:


tstat = reg.coef_/se
pval = stats.t(len(df2)-1).sf(tstat)*2
pval
reg.coef_


# In[7]:


reg2 = linear_model.LinearRegression()
reg2.fit(df2[["Ethnicity_Asian","Ethnicity_Caucasian"]],df2[["Balance"]])


# In[64]:


ssr = ((df2["Balance"].values-reg2.predict(df2[["Ethnicity_Asian","Ethnicity_Caucasian"]]).reshape(1,-1))**2).sum()
sst = ((df2["Balance"].values-df2["Balance"].values.mean())**2).sum()

df_ssr = len(reg2.coef_[0])
df_sst = len(df2)-len(reg2.coef_[0])-1


# In[65]:


fstat = ((sst-ssr)/df_ssr)/(ssr/df_sst)
pval = stats.f(df_ssr,df_sst).sf(fstat)
pval


# In[66]:


reg3 = linear_model.LinearRegression()
reg3.fit(df2[["Student_Yes"]],df2[["Balance"]])


# In[72]:


ssr = ((df2["Balance"].values-reg3.predict(df2[["Student_Yes"]]).reshape(1,-1))**2).sum()
rse = np.sqrt(ssr/(len(df2)-2))
se = rse/(np.sqrt((((df2["Student_Yes"].values-df2["Student_Yes"].mean())**2).sum())))
reg3.coef_


# In[73]:


tstat = reg3.coef_/se
pval = stats.t(len(df2)-1).sf(tstat)*2
pval


# In[74]:


reg4 = linear_model.LinearRegression()
reg4.fit(df2[["Married_Yes"]],df2[["Balance"]])


# In[75]:


ssr = ((df2["Balance"].values-reg4.predict(df2[["Married_Yes"]]).reshape(1,-1))**2).sum()
rse = np.sqrt(ssr/(len(df2)-2))
se = rse/(np.sqrt((((df2["Married_Yes"].values-df2["Married_Yes"].mean())**2).sum())))
reg4.coef_


# In[78]:


tstat = np.abs(reg4.coef_)/se
pval = stats.t(len(df2)-1).sf(tstat)*2
pval


# In[79]:


reg5 = linear_model.LinearRegression()
reg5.fit(df2[["Student_Yes"]],df2[["Income"]])


# In[81]:


ssr = ((df2["Income"].values-reg5.predict(df2[["Student_Yes"]]).reshape(1,-1))**2).sum()
rse = np.sqrt(ssr/(len(df2)-2))
se = rse/(np.sqrt((((df2["Student_Yes"].values-df2["Student_Yes"].mean())**2).sum())))
reg5.coef_


# In[85]:


tstat = np.abs(reg5.coef_)/se
pval = stats.t(len(df2)-1).sf(tstat)*2
pval


# In[ ]:




