from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib
# matplotlib.use('TkAgg') 
import numpy as np 
import pandas as pd 
# import keras
# import tensorflow as tf

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

import os 
USER_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(USER_DIR, 'dev','comp','govhack18','data')
print(os.listdir(DATA_DIR))

def makedir_from_fpath(file_path):
    # https://stackoverflow.com/questions/17057544/python-extract-folder-path-from-file-path
    # print('file_path = ', file_path)
    dir_path = os.path.dirname(file_path)
    # print('dir_path = ', dir_path)
    if not os.path.exists(str(dir_path)):
        print('MAKING dir_path = ', dir_path)
        os.makedirs(str(dir_path))


TMP_DIR = os.path.join(USER_DIR, 'tmp','gcoast')
BEST_MODEL_PATH = os.path.join(TMP_DIR, 'model.h5')
COLUMN_JSON = os.path.join(TMP_DIR, 'columns.json')

TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 4814

INPUT_CSV = 'train_gcoast.csv'
fpath = os.path.join(DATA_DIR, INPUT_CSV)
print('fpath = ', fpath) 

df = pd.read_csv(fpath)
print(df.head())


IGNORE_COLS = ['Unique ID', 'Calendar Year of Insolvency']
df = df.drop(labels = IGNORE_COLS, axis = 1)
print('df.columns = ', df.columns)

# print(df['Sex of Debtor'].unique())
# df = df.replace(['Not Stated'], ['Unknown'])
print(df['Debtor Income'].unique())


df.fillna(0, inplace=True)
print(df.nunique())

df = df.astype('category')
print(df.head())

df = pd.get_dummies(df)
print(df.head())


df = df.drop(labels = ['Y_0'], axis = 1)
    
dfY1 = df[df['Y_1'] == 1]
print('len(dfY1) = ', len(dfY1.index))

dfY0 = df[df['Y_1'] == 0]
print('len(dfY0) = ', len(dfY0.index))
# print(dfY0.head())

from sklearn.model_selection import train_test_split, GridSearchCV
train_df1, val_df1 = train_test_split(dfY1, train_size=TRAIN_VAL_SPLIT,
                                        random_state=RANDOM_STATE)
train_df, val_df = train_test_split(dfY0, train_size=TRAIN_VAL_SPLIT,
                                        random_state=RANDOM_STATE)

train_df = (train_df.append(train_df1)).sample(frac=1)
val_df = (val_df.append(val_df1)).sample(frac=1)
print('len(train_df) = ', len(train_df.index))
print('len(val_df) = ', len(val_df.index))


X_train = train_df.drop(labels = ['Y_1'], axis = 1)
print('X_train.columns.columns = ', X_train.columns)
X_train = X_train.values
makedir_from_fpath(os.path.join(TMP_DIR, 'X_train.npy'))
np.save(os.path.join(TMP_DIR, 'X_train.npy'), np.asarray(X_train))

Y_train = (train_df['Y_1'].values)[..., np.newaxis]
print('X,Y shapes = ', X_train.shape, Y_train.shape)
np.save(os.path.join(TMP_DIR, 'Y_train.npy'), np.asarray(Y_train))

Y_val = (val_df['Y_1'].values)[..., np.newaxis]
val_df = val_df.drop(labels = ['Y_1'], axis = 1)
X_val = val_df.values

import simplejson
with open(COLUMN_JSON, "w") as f:
    print('val_df.columns = ', list(val_df.columns))
    simplejson.dump(list(val_df.columns), f)

print('X,Y shapes = ', X_val.shape, Y_val.shape)
np.save(os.path.join(TMP_DIR, 'X_val.npy'), np.asarray(X_val))
np.save(os.path.join(TMP_DIR, 'Y_val.npy'), np.asarray(Y_val))




    

