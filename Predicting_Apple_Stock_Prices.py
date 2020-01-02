#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math


# In[17]:


df = pd.read_csv("apple.csv")
df.set_index(df["Date"], inplace = True)
df.head()
df.drop("Date", axis = 1, inplace = True)
df.head()


# In[18]:



df["HL_PCT"] = ((df["Adj. High"] - df["Adj. Low"]) / df["Adj. Low"])*100
df["PCT_change"] = ((df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"])*100

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

df.head()


# In[19]:


to_predict = "Adj. Close"
df.isnull().any().any()
number_predictions = 1
number_days_into_account = 1
df["Predicted"] = df[to_predict].shift(-number_days_into_account)
df.head(7)
df.dropna(inplace = True)
df.head()


# In[20]:


train = df.iloc[0:int(math.ceil(0.8*len(df)))]
test = df.iloc[int(math.ceil(0.8*len(df)))::]
train.columns = ["AdjClose","HL_PCT","PCT_change","AdjVolume","Predicted"]
test.columns = ["AdjClose","HL_PCT","PCT_change","AdjVolume","Predicted"]


# In[ ]:





# In[21]:


formula = "Predicted ~ AdjClose + HL_PCT + PCT_change + AdjVolume"
model = smf.ols(formula, data = train)
result = model.fit()


# In[22]:


result.summary()


# In[23]:


test.drop("Predicted", axis = 1, inplace = True)


# In[24]:


test.head()


# In[25]:


result.predict(test)


# In[26]:


df.tail(100)


# In[27]:


#VERSION MEJORADA


# In[105]:


df = pd.read_csv("apple.csv")
df.set_index("Date", inplace = True)
df["HL_PCT"] = ((df["Adj. High"] - df["Adj. Low"]) / df["Adj. Low"])*100
df["PCT_change"] = ((df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"])*100
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]
df.columns = ["AdjClose","HL_PCT", "PCT_change", "AdjVolume"]
df.head()


# In[106]:


past_days = 5
columnas = df.columns.tolist()


# In[107]:


for i in range(past_days):
    for j in columnas:
        df[j+str(i)] = np.nan


# In[108]:


nuevas_columnas = df.columns.tolist()
columnas
del nuevas_columnas[0:4]


# In[109]:


for i in nuevas_columnas:
    df[i] = df[i[:-1]].shift((int(i[-1:])+1))


# In[110]:


df.dropna(inplace = True)


# In[111]:


predictors = df.columns
predictors = predictors.tolist()
del predictors[0:4]


# In[112]:


df2 = df.copy()
columnas_datos = df.columns.tolist()
del columnas_datos[1:4]
df2 = df2[columnas_datos]


# In[114]:


train = df2.iloc[0:int(math.ceil(0.8*len(df2)))]
test = df2.iloc[int(math.ceil(0.8*len(df2)))::]


# In[117]:


formula = "AdjClose ~ AdjClose0"
for i in columnas_datos[2::]:
    formula = formula +"+"+i

model = smf.ols(formula, data = train)
result = model.fit()


# In[118]:


result.summary()


# In[119]:


result.predict(test)


# In[122]:


test.head(15)


# In[ ]:




