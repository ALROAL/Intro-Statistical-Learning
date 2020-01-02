#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Auto.csv")
df = df[["mpg","horsepower"]]
for i in range(len(df["horsepower"].values)):
    if df["horsepower"][i].isnumeric() == False:
        df["horsepower"][i] = np.nan
df.dropna(inplace = True)
df["horsepower"] = pd.to_numeric(df["horsepower"])


# In[3]:


model = smf.ols(formula = "mpg ~ horsepower", data = df)
result = model.fit()


# In[4]:


result.summary()


# In[5]:


fig1 = plt.figure(figsize = (20,10))
axes = fig1.add_subplot(111)

axes.scatter(df["horsepower"],df["mpg"], c = "red", marker = "o")
x = np.linspace(df["horsepower"].values.min(),df["horsepower"].values.max(),50)
y = result.predict(exog=dict(horsepower=x))
axes.plot(x,y)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




