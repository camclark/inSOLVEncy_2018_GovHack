from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib
# matplotlib.use('TkAgg') 
import numpy as np 
import pandas as pd 
import keras
import tensorflow as tf
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

import os
import keras.backend as K
import tensorflow as tf
import simplejson

def make_model(input_shape):
    from keras.models import Model
    from keras.layers import Dense, BatchNormalization, Input, Dropout

    in1 = Input(shape=[input_shape])
    x = in1
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)

    # x = Dense(32, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)

    # x = Dense(32, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(in1, x)
    print(model.summary())
    return model


def main():
    # TRAIN_ON = True
    TRAIN_ON = False

    MODEL = os.path.splitext(os.path.basename(__file__))[0]
    print('MODEL = ', MODEL)
    USER_DIR = os.path.expanduser('~')
    DATA_DIR = os.path.join(USER_DIR, 'dev','comp','govhack18','data')
    print(os.listdir(DATA_DIR))

    TMP_DIR = os.path.join(USER_DIR, 'tmp','gcoast')
    MODEL_DIR = os.path.join(TMP_DIR, MODEL)
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
    makedir_from_fpath(BEST_MODEL_PATH)
    BEST_MODEL_JSON = BEST_MODEL_PATH + '.json'
    COLUMN_JSON = os.path.join(TMP_DIR, 'columns.json')
    COLUMNS_SORTED_PATH = os.path.join(TMP_DIR, 'columns_sorted.csv')
    WEIGHTS_JSON = os.path.join(TMP_DIR, 'weights.npy')

    X_train = np.load(os.path.join(TMP_DIR, 'X_train.npy'))
    Y_train = np.load(os.path.join(TMP_DIR, 'Y_train.npy'))
    print('Y_train.mean, min, max = ', np.mean(Y_train), np.min(Y_train), np.max(Y_train))

    X_val = np.load(os.path.join(TMP_DIR, 'X_val.npy'))
    Y_val = np.load(os.path.join(TMP_DIR, 'Y_val.npy'))
    print('Y_val.mean, min, max = ', np.mean(Y_val), np.min(Y_val), np.max(Y_val))

    # model_loss = 'binary_crossentropy'
    model_compare_metric = 'val_loss'
    model = make_model(X_train.shape[-1])

    if TRAIN_ON:
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
        model.compile(loss='mae', optimizer='adam', metrics=['mae', f1_v2])

        from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.callbacks import TensorBoard
        ''' --> Callbacks '''
        early_stop = EarlyStopping(monitor=model_compare_metric,
                                   min_delta=0,
                                   patience=32, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor=model_compare_metric,
                                      factor=0.5, min_delta=0, verbose=1,
                                      patience=10, cooldown=0)
        tf_board = TensorBoard(log_dir=MODEL_DIR, write_images=False)
        save_best = ModelCheckpoint(monitor=model_compare_metric,
                                    save_weights_only=True,
                                    filepath=BEST_MODEL_PATH, save_best_only=True)

        callback_list = [early_stop, save_best, tf_board, reduce_lr]

        # serialize model to JSON
        model_json = model.to_json()
        with open(BEST_MODEL_JSON, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")
        model.fit(X_train, Y_train, epochs=1000, batch_size=64, validation_data=(X_val, Y_val),
                  callbacks=callback_list)
    else:
        from keras.models import model_from_json
        # load json and create model
        json_file = open(BEST_MODEL_JSON, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print(model.summary())

        # load weights into new model
        model.load_weights(BEST_MODEL_PATH)
        print("Loaded model from disk")

        y_pred = model.predict(X_val)
        y_pred = np.round(y_pred, decimals=0)

        from sklearn import metrics
        cm = metrics.confusion_matrix(Y_val, y_pred)
        print(cm)

        f1 = metrics.f1_score(Y_val, y_pred, average=None)
        print(f1)

        f1 = metrics.f1_score(Y_val, y_pred, average='micro')
        print('f1 micro =', f1)

        acc = metrics.accuracy_score(Y_val, y_pred)
        print('acc =', acc)

        # p = metrics.precision_score(y_gt, y_pred, average='micro')
        # print('precision_score micro =', p)

        p = metrics.precision_score(Y_val, y_pred, average=None)
        print('precision_score =', p)

        layer = model.layers[-1]
        s = layer.get_weights()[0].shape
        print(s)
        w = layer.get_weights()[0]
        print(w)
        print(w.shape)
        w = np.squeeze(w)
        with open(WEIGHTS_JSON, "w") as f:
            # print('val_df.columns = ', list(w))
            np.save(f, w)

        w_abs = np.abs(w)

        print('LOADING labels from file = ', COLUMN_JSON)

        with open(COLUMN_JSON, 'r') as f:
            columns = simplejson.load(f)
            print('LOADED columns = ', columns)

        print(w.shape)
        df = pd.DataFrame({ 'columns' : columns, 'w_abs': w_abs, 'w': w})
        print(df.head())

        df = df.sort_values(by=['w_abs'], ascending=False)
        print(df.head(30))
        print('bias = ', layer.get_weights()[1])

        df.to_csv(COLUMNS_SORTED_PATH, index=False)

        # Plot the total crashes
        # sns.set_color_codes("pastel")
        # sns.barplot(x="total", y="abbrev", data=crashes,
        #             label="Total", color="b")


def makedir_from_fpath(file_path):
    # https://stackoverflow.com/questions/17057544/python-extract-folder-path-from-file-path
    # print('file_path = ', file_path)
    dir_path = os.path.dirname(file_path)
    # print('dir_path = ', dir_path)
    if not os.path.exists(str(dir_path)):
        print('MAKING dir_path = ', dir_path)
        os.makedirs(str(dir_path))


def f1_loss(y_true, y_pred):
    return 1. - f1_v2(y_true, y_pred)


def f1_v2(y_true, y_pred):
    # from keras import backend as K
    # https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
    main()



    

