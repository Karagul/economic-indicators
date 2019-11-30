#!/usr/bin/env python
# coding: utf-8

# In[80]:


######not finish yet****************$$$$$$$$$$$$$$$$$$
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse


# In[81]:


data = pd.read_csv('econ.csv')
data.head()


# In[82]:


corr=data.corr()
plt.figure(figsize=(13,8))
sns.heatmap( data.corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='black',annot=True)


# In[83]:


new_data=data.drop(['Period', 'CompLead', 'BusConf','Emp','InvToSales','PMI','MfgOrdDur'
                    ,'BldgPerm','FedFunds'], axis=1)
new_data.head()


# In[84]:


new_data.info()


# In[85]:


type(new_data)


# In[86]:


new_data.shape


# In[87]:


new_data.columns


# In[88]:


new_data.iloc[30:40,2:3]


# In[89]:


new_data.tail()


# In[90]:


corr=new_data.corr()
sns.heatmap( new_data.corr(),linewidths=0.2,vmax=1.0,square=True,linecolor='red',annot=True)


# In[102]:


new_data.info()


# In[92]:


new_data.head()


# In[93]:


#knn classification
X = new_data.iloc[:,1: ].values
y = new_data.iloc[:, 0].values


# In[98]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[99]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[100]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:




