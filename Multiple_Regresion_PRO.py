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


df1 = pd.read_csv("Credit.csv", index_col = 0)
df1.head()


# In[3]:


df = pd.get_dummies(df1, columns = ["Gender", "Student","Married","Ethnicity"], drop_first = True)
df.head()


# In[4]:


n = len(df)
p = df.columns.tolist()
order = [0,1,2,3,4,5,7,8,9,10,11]
p = [p[i] for i in order]
df2 = df[p]
df2.head()


# In[5]:


suma_p = {}
for i in p:
    if i != "Income":
        suma_p[i] = "+" + i
    else:
        suma_p[i] = i
formula = "Balance ~ "
pevaluados = []
for i in p:
    formula = formula + suma_p[i]
    print(formula)
    model = smf.ols(formula = formula, data = df)
    result = model.fit()
    pevaluados.append(i)
    print(pevaluados)
    print(result.pvalues)
    for j in pevaluados:
        if result.pvalues[j]>=0.08:
            print(suma_p[j])
            print(formula)
            formula = formula.replace(suma_p[j],"")
            pevaluados.remove(j)
            print(formula)
model = smf.ols(formula = formula, data = df)
result = model.fit()


# In[6]:


df_p = df[pevaluados]
x = df_p.loc[1,:]


# In[7]:


model = smf.ols(formula = formula, data = df)
result = model.fit()
#result.predict(df_p)
formula


# In[8]:


result.summary()


# In[9]:


print(formula)
corr_o = df_p.corr()
corr = df_p.corr().values
corr = np.triu(corr)
for i in range(len(pevaluados)):
    for j in range(len(pevaluados)):
        if i != j:
            if corr[i][j]>=0.75:
                formula = formula.replace("+"+pevaluados[i],"")
print(formula)
corr


# In[10]:


corr_o


# In[17]:


model2 = smf.ols(formula = "Balance ~ Income + Limit + Income:Limit", data = df)
result2 = model2.fit()
result2.summary()


# In[ ]:





# In[ ]:




