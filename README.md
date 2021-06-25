# Overview

The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

<h2>Files available :</h2>

<ul>
<li>sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.</li>
<li>test.csv - the test set. You need to forecast the sales for these shops and products for November 2015. </li>
<li>sample_submission.csv - a sample submission file in the correct format. items.csv - supplemental information about the items/products. </li>
<li>item_categories.csv - supplemental information about the items categories. </li>
<li>shops.csv- supplemental information about the shops.</li>
 <ul>


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:20,.2f}'.format
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
```


```python
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor



from sklearn.metrics import mean_squared_error
```


```python
import pickle
```


```python
import xgboost as xgb
```

# Understand the problem


```python
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")


items = pd.read_csv("../input/predict-future-sales-translated-dataset/items_en.csv")
shops = pd.read_csv("../input/predict-future-sales-translated-dataset/shops_en.csv")
categories = pd.read_csv("../input/predict-future-sales-translated-dataset/item_categories_en.csv")

submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
```


```python
train.head()
```


```python
print(' train set  :{} \n test set : {}'.format(list(train.columns), list(test.columns)))
```


```python
print(' shape train {} \n shape test {} \n total data {}'.format(train.shape, test.shape, train.shape[0]+test.shape[0]))
print(" train dataset have null: {} \n test dataset have null: {} ".format(train.isnull().sum().any(), test.isnull().sum().any()))

```

#### Review of same items in train and test set

Here we will can see that the values in the item shop and the shop id not are same, therefore later we going to remove the items that not make match between dataset, it is because is not necesary.


```python
#review shop_id in train set and test set
print(' unique values of shop_id in train set: \n {} \n\n unique values of shop_id in test set: \n {}'.format(len(train.shop_id.unique()),len(test.shop_id.unique())))
```

#### How are the frequencies in the dates


```python
# how are items most freqs 
datesfreq = train.groupby('date_block_num')['shop_id'].count().sort_values(ascending=False).head()
```


```python
dates = train[train.date_block_num.isin(list(datesfreq.index))]
dates = dates.groupby('date')['shop_id'].count().sort_values(ascending=True).reset_index()
dates = dates.set_index('date')
dates.index = pd.to_datetime(dates.index)
dates.columns = ['freq']
```


```python
print('months: {} \n\n days: {}'.format(sorted(list(dates.index.month.unique())) ,sorted(list(dates.index.day.unique()))))
```

In the following figure it is possible to see that the peak value was in December, it's is logical because is an age when there is a lot of movements transaction for christmas, even the percentil 0.95 is alocated between November and January. 


```python
maxvalue = dates.sort_values(by='freq',ascending=False)[:1]
ax = dates.plot(color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Freq')
ax.axvline(maxvalue.index, color='green', linestyle='--', alpha=0.3)
ax.axhline(maxvalue.values, color='blue', linestyle='--', alpha=0.3)
ax.axhline(dates.quantile(0.95).values, color='purple', linestyle='--', alpha=0.3)
ax.text(maxvalue.index,maxvalue.values, '  maxvalue : {} : {}'.format(str(maxvalue.index.date[0]),str(maxvalue.values[0][0])), fontsize=6)
ax.text(maxvalue.index,dates.quantile(0.95).values, ' percentile 0.95', fontsize=6)
plt.show()
```


```python
print('How many item_id are? : {}'.format(len(train.item_id.unique())))
```


```python
train['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7, figsize=(12,3))
plt.title("Shop Id Values")
```


```python
train['item_price'].plot(kind='hist', alpha=0.7, color='orange', figsize=(12,3))
plt.title("Item price Histogram")
```


```python
train['item_cnt_day'].plot(kind='hist', alpha=0.7, color='green', figsize=(12,3))
plt.title('Item count by day Histogram')
```


```python
train['item_cnt_day'].sort_values(ascending=False)[:5]
```


```python
train[train['item_cnt_day'] == 2169]
```

Item 11373 was sold 2169 times at shop 12 on a single day in Octuber.


```python
items[items['item_id'] == 11373]
```


```python
train[train['item_id'] == 11373].head(5)
```


```python
train[train['item_id']== 11373]['item_cnt_day'].median()
```


```python
train[train['item_id']== 11373]['item_cnt_day'].plot(kind='hist',figsize=(12,3))
plt.text(1000,300,'median item_cnt_day: {}'.format(train[train['item_id']== 11373]['item_cnt_day'].median()))
```

Now  it is possible to see that value 2169 in column item_cnt_day is a value anomaly, so we going to impute this value


```python
train = train[train['item_cnt_day'] < 2000]
train['item_price'].sort_values(ascending=False)[:5]
```

##### Price 307.980


```python
train[train['item_price'] == 307980]
```


```python
items[items['item_id'] == 6066]
```


```python
train[train['item_id']== 6066]
```

Radmin 3 to 522 people, it mean thata all price is by 522 people.
It is only item, there is not point benchmark to understand the value, so it will be drop from training set. 


```python
train = train[train['item_price'] < 300000]
```

##### Price -1 


```python
train['item_price'].sort_values()[:5]
```


```python
train[train['item_price'] == -1]
```


```python
items[items['item_id'] == 2973]
```


```python
train[train['item_id'] == 2973].head(5)
```

##### Stores
We going to how much store there are and if they are duplicated


```python
len(train['shop_id'].unique())
```


```python
len(test['shop_id'].unique())
```


```python
shops.T
```

Effectively store are duplicated, so we going to replace id with a id most representative


```python
train.loc[train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57

train.loc[train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

train.loc[train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
```

Also there are store that have as name, the name city, so it could be complicated for separed the same store, so we going to remove name city from store.


```python
shops['city'] = shops['shop_name'].str.split(' ').map(lambda row: row[0])
```


```python
shops[shops['city'] == '!']
```


```python
shops['city'] = shops['city'].str.replace('!','Yakutsk')
```


```python
shops['city'].unique()
```


```python
from sklearn import preprocessing
```

##### Encode Name store 

Now we going to encode name store to make modelling most easier for processing. 


```python
le = preprocessing.LabelEncoder()
le.fit_transform(shops['city'])
```


```python
shops['city_label'] = le.fit_transform(shops['city'])
shops.drop(['shop_name', 'city'], axis=1, inplace=True)
shops.head()
```

### Items_Analysis


```python
items_train = train['item_id'].nunique()
items_test = test['item_id'].nunique()
print('items_train {} \n item_test {}'.format(items_train, items_test))
```


```python
items_train_list = list(train['item_id'].unique())
items_test_list = list(test['item_id'].unique())

flag = 0
if(set(items_test_list).issubset(set(items_train_list))):
    flag=1
if (flag) : 
    print ("Yes, list is subset of other.") 
else : 
    print ("No, list is not subset of other.") 
```

It mean that test df there is value that is not on train set


```python
len(set(items_test_list).difference(items_train_list))
```


```python
categories_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
```


```python
categories.loc[~categories['item_category_id'].isin(categories_in_test)].T
```


```python
le = preprocessing.LabelEncoder()
main_categories = categories['item_category_name'].str.split('-')
categories['main_category_id'] = main_categories.map(lambda row: row[0].strip())
categories['main_category_id'] = le.fit_transform(categories['main_category_id'])
categories['sub_category_id'] = main_categories.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
categories['sub_category_id'] = le.fit_transform(categories['sub_category_id'])
```


```python
categories.head()
```


```python
from itertools import product
```

Testing generation of cartesian product for the month of February in 2013


```python
shops_in_jan = train.loc[train['date_block_num'] == 0,'shop_id'].unique()
items_in_jan = train.loc[train['date_block_num']==0, 'item_id'].unique()
```


```python
jan = list(product(*[shops_in_jan, items_in_jan,[0]]))
```


```python
shops_in_feb = train.loc[train['date_block_num']==1, 'shop_id'].unique()
items_in_feb = train.loc[train['date_block_num']==1, 'item_id'].unique()
feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
```


```python
cartesian_test = []
cartesian_test.append(np.array(jan))
cartesian_test.append(np.array(feb))
```


```python
cartesian_test
```


```python
cartesian_test = np.vstack(cartesian_test)
cartesian_test
```


```python
cartesian_test_df = pd.DataFrame(cartesian_test, columns = ['shop_id', 'item_id', 'date_block_num'])
```


```python
months = train['date_block_num'].unique()
```


```python
from tqdm import tqdm_notebook

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)
    
    return df
```


```python
months = train['date_block_num'].unique()
```


```python
cartesian = []
for month in months:
    shops_in_month = train.loc[train['date_block_num']==month, 'shop_id'].unique()
    items_in_month = train.loc[train['date_block_num']==month, 'item_id'].unique()
    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
```


```python
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
```


```python
x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
x.head()
```


```python
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
```


```python
new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
```


```python
del x
del cartesian_df
del cartesian
del cartesian_test
del cartesian_test_df
del feb
del jan
del items_test_list
del items_train_list
#del train
```


```python
new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
new_train.head()
```


```python
test.insert(loc=3, column='date_block_num', value=34)
```


```python
test['item_cnt_month'] = 0
```


```python
new_train = new_train.append(test.drop('ID', axis = 1))
```


```python
new_train = pd.merge(new_train, shops, on=['shop_id'], how='left')
new_train.head()
```


```python
new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')
new_train.head()
```


```python
new_train = pd.merge(new_train, categories.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')
new_train.head()
```

#### Lag datasets

Now, in this step we going to generate lag over each column


```python
def generate_lag(train, months, lag_column):
    for month in months:
        # Speed up by grabbing only the useful bits
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]
        train_shift['date_block_num'] += month
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train
```


```python
del items
del categories
del shops
del test
```


```python
new_train = downcast_dtypes(new_train)
```


```python
import gc
gc.collect()
```


```python
%%time
new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')
```


```python
%%time
group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')
new_train.drop(['item_month_mean'], axis=1, inplace=True)
```


```python
%%time
group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')
new_train.drop(['shop_month_mean'], axis=1, inplace=True)
```


```python
new_train.columns
```


```python
%%time
group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
new_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')
new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)
```


```python
%%time
group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'main_category_month_mean')
new_train.drop(['main_category_month_mean'], axis=1, inplace=True)


```


```python
%%time
group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'sub_category_month_mean')
new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
```


```python
new_train.head()
```


```python
new_train['month'] = new_train['date_block_num'] % 12
```

Add Holiday days to ddataset


```python
holiday_dict = {
    0: 6,
    1: 3,
    2: 2,
    3: 8,
    4: 3,
    5: 3,
    6: 2,
    7: 8,
    8: 4,
    9: 8,
    10: 5,
    11: 4,
}
```


```python
new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)
```


```python
moex = {
    12: 659, 13: 640, 14: 1231,
    15: 881, 16: 764, 17: 663,
    18: 743, 19: 627, 20: 692,
    21: 736, 22: 680, 23: 1092,
    24: 657, 25: 863, 26: 720,
    27: 819, 28: 574, 29: 568,
    30: 633, 31: 658, 32: 611,
    33: 770, 34: 723,
}
```


```python
new_train['moex_value'] = new_train.date_block_num.map(moex)
```


```python
new_train = downcast_dtypes(new_train)

```


```python
import xgboost as xgb
```


```python
new_train = new_train[new_train.date_block_num > 11]
```


```python
import gc
gc.collect()
```


```python
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace=True)         
    return df

new_train = fill_na(new_train)
```


```python
def xgtrain():
    regressor = xgb.XGBRegressor(n_estimators = 5000,
                                 learning_rate = 0.01,
                                 max_depth = 10,
                                 subsample = 0.5,
                                 colsample_bytree = 0.5)
    
    regressor_ = regressor.fit(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 
                               new_train[new_train.date_block_num < 33]['item_cnt_month'].values, 
                               eval_metric = 'rmse', 
                               eval_set = [(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 
                                            new_train[new_train.date_block_num < 33]['item_cnt_month'].values), 
                                           (new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1).values, 
                                            new_train[new_train.date_block_num == 33]['item_cnt_month'].values)
                                          ], 
                               verbose=True,
                               early_stopping_rounds = 50,
                              )
    return regressor_
```


```python
%%time
regressor_ = xgtrain()
```


```python
predictions = regressor_.predict(new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis = 1).values)
```


```python
regressor_.save_model("model.json")
```


```python
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27

cols = new_train.drop('item_cnt_month', axis = 1).columns
plt.barh(cols, regressor_.feature_importances_)
plt.show()
```


```python
submission['item_cnt_month'] = predictions
```


```python
submission.to_csv('sales_faster_learn.csv', index=False)
```


```python
sub_df.to_csv('Submission_3.csv',index=False)
```
