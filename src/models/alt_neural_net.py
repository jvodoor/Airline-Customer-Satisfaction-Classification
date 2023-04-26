import sys
import path 

#path to add in directories to call other files and their functions
sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']

import make_dataset as md 
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import os


def neural_net_train(X_train, X_test, y_train, y_test, n_hidden_units, n_hidden_layers):
    y = x = keras.layers.Input(shape=X_train.shape[1:])
    y = keras.layers.Dense(n_hidden_units)(y)
    for _ in range(n_hidden_layers):
        y_resid = y
        y = keras.layers.LayerNormalization()(y)
        y = keras.layers.Dense(n_hidden_units,
                                activation=keras.activations.relu)(y)
        
        y = keras.layers.Add()([y,y_resid])
        
    y = keras.layers.Dense(len(np.unique(y_train)),
                            activation=keras.activations.softmax)(y)
    model = keras.Model(x,y)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
                    loss=keras.losses.SparseCategoricalCrossentropy(), 
                     metrics=keras.metrics.SparseCategoricalAccuracy())
            
    keras.utils.plot_model(model,show_shapes=True,expand_nested=True) #will show up as a png file
    return model 

def evaluate_model (nn_model, X_train, y_train, X_test, y_test, n_epochs, num_batch):
    keras.utils.plot_model(nn_model,to_file = 'model2.png', show_shapes=True,expand_nested=True)
    nn_history = nn_model.fit(X_train, y_train,
                                                    epochs=n_epochs,
                                                    verbose=1,
                                                    batch_size=num_batch,
                                                    validation_data=(X_test,y_test))
    print(nn_model.evaluate(X_test,y_test))

#opening file paths for users to load in datasets. File path points to default home directory of project folder.
#Home directory = "Data-4950-Capstone" project folder. If your datasets are in another folder, you must modify this
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', '..')


#not using a dataframe in a neural net, but using this with existing functions to get our separated data
df_train = md.load_dataset(file_path + "/data/processed/df feat eng done.csv")


#split dataset into X df featuring every column but target, and y which is just the target
X, y = md.x_y_split(df_train, -1)

#do the same for our neural net array, except in numpy array form
nn_array_X = X.to_numpy()
nn_array_y = y.to_numpy()


#numpy array 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(nn_array_X, nn_array_y, test_size = 0.2, random_state = 21)

nn_model = neural_net_train(X_train, X_test, y_train, y_test, 64, 1)

nn_model.summary()


evaluate_model (nn_model, X_train, y_train, X_test, y_test, 10, 256)


'''
With anywhere from 45-60% validation accuracy depending on hyper parameter tuning, the Neural Network model falls
well short of the 88%+ accuracy numbers of a logistic regression. 

Ultimately, a deep residual neural network can't produce the results that more simplistic models like logistic regression can.
this is probably because of two reasons:

1. The dataset is fairly simplistic in nature, and in general, a neural net with a dataset that's too simplistic
can sometimes struggle with overfitting and producing good results. 

2. Tabular, or the categorical data present throghout my dataset, is not ideal for neural networks. It can be done,
but it would require a lot of research and analysis of fairly experimental methods, as opposed to methods that 
are more well defined in nature. This is out of the scope of my knowledge, but I do hope to one day return and
try to use some of those methods I discovered.

I will leave this code in as proof of concept for a neural net architecture, but it will not serve as an alternative model.
'''