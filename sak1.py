#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# // we will build a deep learning model to predict the attrition and use precision to measure performance of our model //

df = pd.read_csv("C:\\Users\\HP\\Downloads\\customer_attrition_train.csv")
df.sample(50)
print(df)


# In[2]:


df.drop('ID',axis= 'columns',inplace = True)
print(df)
print(df.dtypes)


# In[3]:


print(df.GrandPayment.values)
pd.to_numeric(df.GrandPayment , errors='coerce').isnull


# In[4]:


def print_unique_col_values(df):
    for column in df :
        if df [column].dtypes=='object':
            print (f'{column}:{df[column].unique()}')
print_unique_col_values(df)


# In[5]:


columnys = ['Aged', 'Married', 'TotalDependents','MobileService','CyberProtection','HardwareSupport','TechnicalAssistance','FilmSubscription','CustomerAttrition']

for items in columnys:
    df[items].replace({'No':0,'Yes':1}, inplace = True)
# df['TotalDependents'].replace({'No':0,'Yes':1},inplace = True)
df['sex'].replace({'Female':0,'Male':1}, inplace=True)
df = pd.get_dummies(data=df , columns = ['4GService','SettlementProcess'])


# In[6]:


for col in df:
    print(f'{col}:{df[col].unique()}')
df


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols_to_scale = ['ServiceSpan', 'QuarterlyPayment','GrandPayment']
df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])


# In[10]:


for col in df:
    print(f'{col}:{df[col].unique()}')


# In[11]:


x=df.drop('CustomerAttrition',axis='columns')
y=df['CustomerAttrition']


# In[71]:


from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split (x,y,test_size=0.0378,random_state=8)
#0.3483
x_train[:25]
len(x_train.columns)


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# In[33]:


import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
    keras.layers.Dense(16,input_shape=(19,),activation='relu'),
    keras.layers.Dense(12 , activation = 'relu'),
    keras.layers.Dense(1,activation='sigmoid')
]
)
# model.summary()
model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train,y_train, epochs=20, batch_size=150, shuffle=True)

# model.predict()


# In[72]:


model.evaluate(x_test,y_test)


# In[28]:


model.predict(x_test)


# In[ ]:




