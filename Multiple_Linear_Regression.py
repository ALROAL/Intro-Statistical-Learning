#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


# In[2]:


df = pd.read_csv("Advertising.csv", index_col = 0)


# In[3]:


df.head()


# In[4]:


reg = linear_model.LinearRegression()
reg.fit(df[["TV", "Radio", "Newspaper"]],df[["Sales"]])
reg.coef_
reg.intercept_
reg.coef_.shape[1]


# In[5]:


ssr = ((df["Sales"].values.reshape(-1,1)-reg.predict(df[["TV", "Radio", "Newspaper"]]))**2).sum()
sst = ((df["Sales"].values-df["Sales"].values.mean())**2).sum()

fstat = ((sst-ssr)/reg.coef_.shape[1])/(ssr/(len(df)-reg.coef_.shape[1]-1))
fstat

pval = stats.f(reg.coef_.shape[1], (len(df)-reg.coef_.shape[1]-1)).sf(fstat)
pval


# In[6]:


reg = linear_model.LinearRegression()
reg.fit(df[["TV", "Newspaper"]], df[["Sales"]])



# In[14]:


import statsmodels.formula.api as smf
model = smf.ols(formula = 'Sales ~ TV + Newspaper', data = df)
results_formula = model.fit()
results_formula.params

xsurf, ysurf = np.meshgrid(np.linspace(df["TV"].values.min(),df["TV"].values.max(),100), np.linspace(df["Newspaper"].values.min(),df["Newspaper"].values.max(),100))
xy = pd.DataFrame({"TV": xsurf.ravel(), "Newspaper": ysurf.ravel()})

fittedy = results_formula.predict(xy)

print(fittedy.shape)

fittedy = np.array(fittedy)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["TV"], df["Newspaper"], df["Sales"], c = "red", marker = "o", alpha = 0.5)
ax.plot_surface(xsurf,ysurf,fittedy.reshape(xsurf.shape), color = "None", alpha = 0.3)
plt.show()
fittedy.reshape(xsurf.shape).shape


# In[13]:


from matplotlib import cm

get_ipython().run_line_magic('matplotlib', '')

xsurf, ysurf = np.meshgrid(np.linspace(df["TV"].values.min(),df["TV"].values.max(),100), np.linspace(df["Newspaper"].values.min(),df["Newspaper"].values.max(),100))
xy = pd.DataFrame({"TV": xsurf.ravel(), "Newspaper": ysurf.ravel()})

z = reg.predict(xy).reshape(xsurf.shape)

fig = plt.figure(figsize = (10,10))
graph1 = fig.add_subplot(111, projection = "3d")

graph1.scatter(df["TV"], df["Newspaper"], df["Sales"], c = "red", marker = "o", alpha = 0.5)
graph1.plot_surface(xsurf,ysurf,z, cmap=cm.coolwarm)

df2 = df[["TV","Newspaper","Sales"]]
pts = df2.values

z = reg.predict(df[["TV", "Newspaper"]])

df3 = df2.copy()
df3["Sales"] = z

pts_plano = df3.values

for i in range(200):
        graph1.plot([pts[i][0], pts_plano[i][0]], [pts[i][1], pts_plano[i][1]],zs=[pts[i][2], pts_plano[i][2]], c = "black")
plt.show()


# In[15]:


xpt = df["TV"].values[0]
ypt = df["Newspaper"].values[0]

xypt = np.array([[xpt,ypt],])

zpt = reg.predict(xypt)

xypt = np.array([[xpt,ypt],])
xpt,ypt,zpt


# In[17]:


df2 = df[["TV","Newspaper","Sales"]]
pts = df2.values

z = reg.predict(df[["TV", "Newspaper"]])

df3 = df2.copy()
df3["Sales"] = z

pts_plano = df3.values

for i in range(200):
        graph1.plot([pts[i][0], pts_plano[i][0]], [pts[i][1], pts_plano[i][1]],zs=[pts[i][2], pts_plano[i][2]], c = "black")


# In[ ]:





# In[ ]:




