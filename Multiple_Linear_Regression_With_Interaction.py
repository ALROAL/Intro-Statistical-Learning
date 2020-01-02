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


# In[7]:


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(interaction_only=True,include_bias = False)
x = poly.fit_transform(df[["TV", "Radio"]])

reg2 = linear_model.LinearRegression()
reg2.fit(x, df["Sales"])

reg2.coef_,reg2.intercept_


# In[12]:


from matplotlib import cm
get_ipython().run_line_magic('matplotlib', '')

xsurf, ysurf = np.meshgrid(np.linspace(df["TV"].values.min(),df["TV"].values.max(),100), np.linspace(df["Radio"].values.min(),df["Radio"].values.max(),100))
interaction = xsurf*ysurf
xy = pd.DataFrame({"TV": xsurf.ravel(), "Radio": ysurf.ravel(), "Interaction": interaction.ravel()})

z = reg2.predict(xy).reshape(xsurf.shape)

fig = plt.figure(figsize = (10,10))
graph1 = fig.add_subplot(111, projection = "3d")


plt.xlabel("TV")
plt.ylabel("Radio")
graph1.scatter(df["TV"], df["Radio"], df["Sales"], c = "red", marker = "o", alpha = 0.5)
graph1.plot_surface(xsurf,ysurf,z,cmap=cm.coolwarm)


df2 = df[["TV","Radio","Sales"]]
pts = df2.values

interaction2 = df["TV"].values*df["Radio"].values

df3 = df[["TV","Radio"]]
df5 = df3.copy()
df5["Interaction"] = interaction2

z_plano = reg2.predict(df5)

df4 = df2.copy()
df4["Sales"] = z_plano

pts_plano = df4.values

for i in range(200):
    graph1.plot([pts[i][0], pts_plano[i][0]], [pts[i][1], pts_plano[i][1]],zs=[pts[i][2], pts_plano[i][2]], c = "black")
plt.show()


# In[ ]:




