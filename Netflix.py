#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:\\Users\\sudhi\\Downloads\\titles.csv")
df


# In[5]:


df.info


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df1 = df.dropna(subset=["title","description"])


# In[11]:


df1= df1.drop(["seasons"],axis=1)


# In[12]:


df1


# In[13]:


df1.isnull().sum()


# In[14]:


df1.shape


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


df1.columns


# In[17]:


le = LabelEncoder()


# In[18]:


df1.fillna(method="bfill",inplace=True)


# In[19]:


df1["type"]=le.fit_transform(df1["type"])
df1["age_certification"]= le.fit_transform(df1["age_certification"])


# In[20]:


df1


# In[21]:


df1.columns


# In[22]:


df1.isnull().sum()


# In[23]:


x = df1.drop(['type', 'title' , 'description','imdb_id', 'imdb_score', 'imdb_votes',
       'tmdb_score', 'production_countries','id','genres'],axis=1)
x


# In[24]:


y = df1["type"]
y


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[27]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[28]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[29]:


x_train = scale.fit_transform(x_train)


# In[30]:


x_test = scale.transform(x_test)


# In[31]:


model.fit(x_train,y_train)


# In[32]:


y_pred = model.predict(x_test)


# In[33]:


np.unique(y_pred,return_counts=True)


# In[34]:


np.unique(y_test,return_counts=True)


# In[35]:


plt.plot(y_test,y_pred)


# In[36]:


df2= pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df2


# In[37]:


df2.plot(figsize=(15,8))


# In[38]:


df2.plot(figsize=(15,15),kind='bar')


# In[40]:


import seaborn as sns
df1.corr()


# In[42]:


sns.regplot(x="Actual",y="Predicted",data=df2,logistic=True)


# In[48]:


from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_absolute_error
accuracy_score(y_test,y_pred)


# In[50]:


f1_score(y_test,y_pred)


# In[51]:


mean_squared_error(y_test,y_pred)


# In[52]:


mean_absolute_error(y_test,y_pred)


# In[53]:


sns.heatmap(df.corr(),annot=True)


# In[54]:


sns.regplot(x="Actual",y="Predicted",data=df2)


# In[55]:


from sklearn.metrics import confusion_matrix


# In[56]:


confusion_matrix(y_test,y_pred)

