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

TMP_DIR = os.path.join(USER_DIR, 'tmp','gh')
BEST_MODEL_PATH = os.path.join(TMP_DIR, 'model.h5')
COLUMN_JSON = os.path.join(TMP_DIR, 'columns.json')

TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 4814

INPUT_CSV = 'train_8_1500.csv'
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


# np.savetxt(os.path.join(TMP_DIR, 'X_train.csv'), np.asarray(X_train, dtype=int), delimiter=",")
# np.savetxt(os.path.join(TMP_DIR, 'Y_train.csv'), np.asarray(Y_train, dtype=int), delimiter=",")

# corr = df.corr()
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

# def make_model(input_shape):
#     from keras.models import Model
#     from keras.layers import Dense, BatchNormalization, Input, Dropout
#
#     in1 = Input(shape=[input_shape])
#     x = in1
#     # x = Dense(64, activation='relu')(x)
#     # x = Dropout(0.5)(x)
#     # x = BatchNormalization()(x)
#
#     # x = Dense(32, activation='relu')(x)
#     # # x = Dropout(0.5)(x)
#     # x = BatchNormalization()(x)
#
#     # x = Dense(32, activation='relu')(x)
#     # # x = Dropout(0.5)(x)
#     # x = BatchNormalization()(x)
#
#     x = Dense(1, activation='sigmoid')(x)
#     model = Model(in1, x)
#     print(model.summary())
#     return model
#
#
# # def top1acc(y_true, y_pred):
# #     return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)
# # _POS_WEIGHT = None
# # def dk2_weighted_binary_crossentropy(y_true, y_pred):
# #     return K.mean(tf_weighted_binary_crossentropy(y_true, y_pred, pos_weight=_POS_WEIGHT), axis=-1)
#
# # model_loss = 'binary_crossentropy'
# model_compare_metric = 'val_loss'
# model = make_model(X_train.shape[-1])
#
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
#
# from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.callbacks import TensorBoard
# ''' --> Callbacks '''
# early_stop = EarlyStopping(monitor=model_compare_metric,
#                            min_delta=0,
#                            patience=32, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor=model_compare_metric,
#                               factor=0.5, min_delta=0, verbose=1,
#                               patience=10, cooldown=0)
# tf_board = TensorBoard(log_dir=TMP_DIR, write_images=False)
# save_best = ModelCheckpoint(monitor=model_compare_metric,
#                                  save_weights_only=True,
#                                  filepath=BEST_MODEL_PATH, save_best_only=True)
#
# callback_list = [early_stop, save_best, tf_board, reduce_lr]
#
# model.fit(X_train, Y_train, epochs=1000, batch_size=64, validation_data=(X_val, Y_val),
#           callbacks=callback_list)
#
#


# def make_model():
#     import xgboost as xgb
#         print("Parameter optimization")
#     xgb_model = xgb.XGBClassifier()  # xgb_model = xgb.XGBRegressor()
#     print('X.shape = ', X.shape)
#     print('y.shape = ', y.shape)
#     clf = GridSearchCV(xgb_model,
#                        {'max_depth': [3, 4, 5],
#                         'n_estimators': [30, 40, 50]
#                         # , 'reg_alpha': [0.003, 0.001, 0.0003]
#                         },
#                     n_jobs=12, verbose=1)
#     clf.fit(X, y)
#     print(clf.best_score_)
#     print(clf.best_params_)
    

