#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('C:/Users/lenovo/Downloads/titanic_data.csv')


# In[4]:


df.shape


# In[5]:


data = pd.read_csv('C:/Users/lenovo/Downloads/titanic_data.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.describe(include = "all")


# In[9]:


print (pd.isnull(df).sum())


# In[10]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts())

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts())


# In[ ]:





# In[12]:


sns.barplot(x="Sex", y="Survived", data=df)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# In[21]:


sns.barplot(x="Pclass" , y="Survived" , data=df )
#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)




# In[25]:


sns.barplot(x="SibSp" , y="Survived" , data=df)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 3].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 4].value_counts(normalize = True)[1]*100)


# In[26]:


sns.barplot(x="Parch", y="Survived" , data=df)


# In[28]:


df["CabinBool"] = (df["Cabin"].notnull().astype('int'))

print("percentage 1 survived:" , df["Survived"][df["CabinBool"] == 1].value_counts(normalize = True)[1]*100 )

print("percentage 2 survived:" , df["Survived"][df["CabinBool"] == 0].value_counts(normalize = True)[1]*100 )




sns.barplot(x="CabinBool", y="Survived" , data=df)

            
            
            
            
            
            


# In[29]:


data.shape


# In[30]:


data = data.dropna()


# In[31]:




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing




# In[32]:


data = data.drop(['Embarked', 'Name', 'Cabin', 'Ticket'] , axis = 1)


# In[34]:


data['Sex'] = data['Sex'].map({'male': 0,'female': 1})


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Survived'],axis=1), 
                                                    data['Survived'], test_size=0.20, 
                                                    random_state=8)


# In[36]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[37]:


predictions = logmodel.predict(X_test)
X_test.head()


# In[38]:


accuracy = logmodel.score(X_test,y_test)
print(accuracy*100,'%')


# In[39]:


predictions


# In[40]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:




