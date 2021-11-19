#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#BY Arif Tanzer KIRAR   ArifTanzer@hotmail.com


# In[2]:


df = pd.read_csv(r"C:\Users\PC\Desktop\Kişisel\BTK\Python ile Makine Öğrenmesi\maaslar_yeni.csv")


# In[3]:


df


# In[22]:


sonuc = pd.DataFrame(data=df, index = range(30), columns = ["UnvanSeviyesi","Kidem","Puan"]) 
t=df["maas"]


# In[23]:


t2=t.values.reshape(-1, 1)


# In[24]:


from sklearn.linear_model import LinearRegression # bu aslında polinmial değil ama sadece
#veri seti linear reg'de nasıl çalışıyor görmek isyiyorum

re = LinearRegression()
re.fit(sonuc,t2)  # x'ten y'i öğren


# In[27]:


re.predict(sonuc)


# In[28]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(sonuc)
x_poly


# In[29]:


re2 = LinearRegression()
re2.fit(x_poly, t2)


# In[32]:


re2.predict(poly_reg.fit_transform([[6.6, 5.4, 55]]))


# In[ ]:




