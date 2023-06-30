#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[6]:


heart=pd.read_csv('heart.csv')


# In[7]:


heart.head()


# In[9]:


sns.displot(heart['age'])
plt.show()


# In[10]:


sns.distplot(heart['age'])
plt.show()


# In[12]:


sns.histplot(heart['age'])
plt.show()


# In[13]:


heart.shape


# In[14]:


heart.head()


# In[15]:


heart.describe()


# In[16]:


heart['sex'].value_counts()


# In[19]:


heart['sex'].value_counts().keys()


# In[20]:


heart['sex'].value_counts().values


# In[25]:


plt.bar(list(heart['sex'].value_counts().keys()),list(heart['sex'].value_counts().values),color=["Red","Green"])
plt.show()       


# In[34]:


plt.bar(list(("Male","Female")),list(heart['sex'].value_counts()),color=["Blue","Pink"])
plt.show()


# In[35]:


heart.head()


# In[36]:


heart['cp'].value_counts()


# In[37]:


heart['cp'].value_counts().keys()


# In[38]:


heart['cp'].value_counts().values


# In[39]:


plt.bar(list(heart['cp'].value_counts().keys()),list(heart['cp'].value_counts().values),color=["green","yellow","orange","red"])
plt.show()


# In[44]:


plt.bar(list(("level-0","level-2","level-1","level-3")),list(heart['cp'].value_counts()),color=["green","orange","yellow","red"])
plt.show()


# In[47]:


heart.head()


# In[48]:


sns.distplot(heart['chol'])
plt.show


# In[49]:


sns.displot(heart['chol'])
plt.show


# In[50]:


heart['target'].value_counts()


# In[51]:


heart['cp'].value_counts().keys()


# In[52]:


heart['cp'].value_counts().values


# In[53]:


heart['target'].value_counts().keys()


# In[54]:


heart['target'].value_counts().values


# In[55]:


plt.bar(list(heart['target'].value_counts().keys()),list(heart['target'].value_counts().values),color=["green","red"])
plt.show()


# In[56]:


plt.bar(list(("yes-1","no-0")),list(heart['target'].value_counts()),color=["green","red"])
plt.show()


# In[62]:


plt.bar(list(("not-safe","safe")),list(heart['target'].value_counts()),color=["red","green"])
plt.show()


# In[65]:


x=heart[['age']]
y=heart[['target']]


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[68]:


from sklearn.naive_bayes import MultinomialNB


# In[69]:


mnb = MultinomialNB()


# In[70]:


mnb.fit(x_train,y_train)


# In[71]:


y_pred=mnb.predict(x_test)


# In[74]:


y_test.head(),y_pred[0:5]


# In[79]:


from sklearn.metrics import confusion_matrix


# In[101]:


confusion_matrix = (y_test,y_pred)


# In[102]:


heart.head()


# In[103]:


x=heart[['age','trestbps','chol']]


# In[105]:


x.head()


# In[107]:


y=heart[['target']]


# In[108]:


from sklearn.model_selection import train_test_split


# In[111]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)


# In[113]:


from sklearn.naive_bayes import GaussianNB


# In[114]:


gnb=GaussianNB()


# In[115]:


gnb.fit(x_train,y_train)


# In[118]:


y_pred=gnb.predict(x_test)


# In[119]:


from sklearn.metrics import confusion_matrix


# In[120]:


confusion_matrix(y_test,y_pred)


# In[121]:


(128+124)/(128+124+92+66)


# In[ ]:




