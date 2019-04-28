
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV


# In[49]:


train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")
test2=pd.read_csv("./test.csv")
len_train=train.shape[0]
houses=pd.concat([train,test], sort=False)
print(train.shape)
print(test.shape)


# In[51]:


houses.select_dtypes(include='object').head()


# Incomplete data:
#     
# LotFrontage;
# Alley;
# MasVnrType;
# MasVnrArea;
# BsmtQual        1423
# BsmtCond        1423
# BsmtExposure    1422
# BsmtFinType1    1423
# BsmtFinType2    1422
# Electrical      1459
# FireplaceQu       770
# GarageType       1379
# GarageYrBlt      1379
# GarageFinish     1379
# GarageQual       1379
# GarageCond       1379
# Fence             281
# MiscFeature        54
# PoolQC           7

# In[52]:


houses.select_dtypes(include=['float','int']).head()


# In[53]:


houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]


# In[54]:


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')


# In[55]:


for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(data[col].mode()[0])


# In[56]:


data.select_dtypes(include=['int','float']).isnull().sum()[data.select_dtypes(include=['int','float']).isnull().sum()>0]


# In[57]:


houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]


# "Some NAs means "None" (which I will fill with 0) or means "Not Available" (which I will fill with mean)"

# In[58]:


for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)


# In[59]:


train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())


# 1.4 - Remove some features high correlated and outliers¶

# In[60]:


plt.figure(figsize=[30,15])
sns.heatmap(train.corr(), annot=True)


# In[61]:


#from 2 features high correlated, removing the less correlated with SalePrice
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)


# In[62]:


#removing outliers recomended by author
train = train[train['GrLivArea']<4000]


# In[63]:


houses=pd.concat([train,test], sort=False)


# ### 1.5 Transformation

# In[64]:


# Numerical to categorical

houses['MSSubClass']=houses['MSSubClass'].astype(str)


# In[65]:


# Adjust skew

skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]

train=houses[:len_train]
test=houses[len_train:]


# In[69]:


train.to_csv("train_processed.csv")
test.to_csv("test_processed.csv")

