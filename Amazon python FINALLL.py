#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as pyplot
import seaborn as sns


# In[3]:


Amazon = pd.read_csv("Amazon.csv")


# In[5]:


Amazon.head()


# In[7]:


Amazon.columns


# In[9]:


Amazon.info()


# In[15]:


Amazon['Order Date'] = pd.to_datetime(Amazon['Order Date'])
Amazon['Ship Date'] = pd.to_datetime(Amazon['Ship Date'])


# In[17]:


Amazon['Region'] = Amazon['Region'].astype(str)
Amazon['Country'] =Amazon['Country'].astype(str)
Amazon['Item Type'] = Amazon['Item Type'].astype(str)
Amazon['Sales Channel'] =Amazon['Sales Channel'].astype(str)
Amazon['Order Priority'] = Amazon['Order Priority'].astype(str)
     


# In[19]:


Amazon[['Units Sold', 'Unit Price',	'Unit Cost', 'Total Revenue', 'Total Cost',	'Total Profit']].describe()
     


# In[23]:


Amazon['Order Month'] = Amazon['Order Date'].dt.month
Amazon['Order Year'] = Amazon['Order Date'].dt.year
Amazon['Order Date MonthYear'] =Amazon['Order Date'].dt.strftime('%Y-%m')
Amazon = Amazon.drop(columns=['Order Date'])
     


# In[27]:


df=Amazon


# In[29]:


df.isnull().sum()


# In[31]:


pd.set_option('display.max_rows', None)
df['Country'].value_counts()


# In[41]:


import matplotlib.pyplot as plt 
country_names = df.Country.value_counts().index
country_val = df.Country.value_counts().values
# Pie Chart for top 20 country
fig,ax = plt.subplots(figsize=(8,8))
ax.pie(country_val[:20],labels=country_names[:20],autopct='%1.2f%%')
plt.show()


# In[51]:


sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 2))
sns.boxplot(Amazon['Total Profit'], color="pink", width=.6)

plt.title('Total Profit Boxplot', fontsize=13)
plt.xlabel('Profit')
plt.show()


# In[53]:


def detect_outliers(dataframe, column):
    threshold = 2     ## 2rd standard deviation
    mean = np.mean(column)
    std = np.std(column)
    outliers = []

    for i, value in enumerate(column):
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
            print(dataframe.loc[i])

    return outliers
     


# In[55]:


outliers = detect_outliers(df, df["Total Profit"])


# In[57]:


print(outliers)


# In[59]:


list_length = len(outliers)


# In[61]:


print("The list has", list_length, "outliers in Total Profit column of dataframe data ")


# In[65]:


sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 2))
sns.boxplot(Amazon['Total Cost'], color="plum", width=.6)

plt.title('Total Cost Boxplot', fontsize=13)
plt.xlabel('Cost')
plt.show()


# In[67]:


def detect_outliers(dataframe, column):
    threshold = 2     ## 3rd standard deviation
    mean = np.mean(column)
    std = np.std(column)
    outliers = []

    for i, value in enumerate(column):
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
            print(dataframe.loc[i])

    return outliers


# In[69]:


outliers = detect_outliers(df, df["Total Cost"])
     


# In[71]:


plt.bar(df['Order Month'], df['Total Revenue'])

# Set the chart title and axis labels
plt.title('Number of Orders Purchased by Month and Year')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.xlabel('Order Month')
plt.ylabel('Total Revenue')

# Rotate the x-axis labels for better readability

# Display the chart
plt.show()
     


# In[73]:


df.groupby('Order Year')['Total Profit'].mean().plot()
plt.xlabel('Order Year')
plt.ylabel('Total Profit')
plt.title('Profit per year')


# In[75]:


revenue_by_category = df.groupby('Item Type')['Total Revenue'].sum().sort_values(ascending=False)
revenue_by_category


# In[77]:


profit_by_category = df.groupby('Item Type')['Total Profit'].sum().sort_values(ascending=False)
profit_by_category


# In[79]:


print(df[['Total Revenue', 'Total Cost', 'Total Profit']].corr())


# In[81]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
df["Item Type"] = le.fit_transform(df["Item Type"])
df["Sales Channel"] = le.fit_transform(df["Sales Channel"])
df["Order Priority"] = le.fit_transform(df["Order Priority"])


# In[83]:


df = df.drop("Region", axis=1)
df = df.drop("Country", axis=1)
df = df.drop("Order Date MonthYear", axis=1)
df = df.drop("Order ID", axis=1)
df = df.drop("Ship Date", axis=1)
     


# In[85]:


df.head()


# In[ ]:




