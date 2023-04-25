import sys
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']
import make_dataset as md
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt
import train_model as tm
import pickle

def log_reg_test_data_statistics(logistic, X_test, y_test):

    log_pred_test = logistic.predict(X_test)

    # returns the probability for both class labels
    logexport_prob = logistic.predict_proba(X_test) 

    #print confusion matrix for testing data
    print("Logistic Regression Model Testing Data Confusion Matrix :")
    print(confusion_matrix(y_test, log_pred_test)) 

    #scores
    accuracy_testing = logistic.score(X_test, y_test)
    precision_test = precision_score(y_test, log_pred_test)
    recall_test = recall_score(y_test, log_pred_test)
    print('\n')
    print("Recall: ", recall_test)
    print("Precision: ", precision_test)
    print('Accuracy:', accuracy_testing.round(3))


    # plot ROC curve
    y_pred_prob = logistic.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot([0, 1], [0, 1],'k--')
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree ROC Curve')
    plt.show();

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, logistic.predict(X_test))

    # roc auc score
    log_roc_auc = roc_auc_score(y_test, y_pred_prob)
    roc_auc_format = 'ROC AUC Score: {0:.4f}'.format(log_roc_auc)
    print(roc_auc_format)


#load data set
df_test = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df alt no gender.csv")

#split data into Y target and X predictors
X, y = md.x_y_split(df_test, -1)

#80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

#logistic regression training
logistic_model = tm.log_or_boost_train(X_train, X_test, y_train, 'log')
#logistic regression test statistics
log_reg_test_data_statistics(logistic_model, X_test, y_test) 

#boosted tree training
tree_model = tm.log_or_boost_train(X_train, X_test, y_train, 'boost')
#boosted tree statistics
log_reg_test_data_statistics(tree_model, X_test, y_test) 


