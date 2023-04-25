import sys
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\models']
import make_dataset as md
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import train_model as tm
import pickle as pkl

def log_reg_test_data_statistics(logistic, X_test, y_test, model):

    log_pred_test = logistic.predict(X_test)

    # returns the probability for both class labels
    logexport_prob = logistic.predict_proba(X_test) 

    #print confusion matrix for testing data
    print(model, " Model Testing Data Confusion Matrix :")
    print(confusion_matrix(y_test, log_pred_test)) 

    #scores
    accuracy_testing = logistic.score(X_test, y_test)
    print('\n')

    print (model, "Model performance scores: ", "\n")
    
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
df_final_log_data = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/final data log model.csv")
df_final_tree_data = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/final data tree model.csv")

#split data into Y target and X predictors
log_X, log_y = md.x_y_split(df_final_log_data, -1)
tree_X, tree_y = md.x_y_split(df_final_tree_data, -1)


#80/20 train test split
log_X_train, log_X_test, log_y_train, log_y_test = train_test_split(log_X, log_y, test_size = 0.2, random_state = 21)
tree_X_train, tree_X_test, tree_y_train, tree_y_test = train_test_split(tree_X, tree_y, test_size = 0.2, random_state = 21)

#load picked models
logistic_model = pkl.load(open('C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/models/logistic_train.pkl', 'rb'))
tree_model = pkl.load(open('C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/models/xg_train.pkl', 'rb'))

#logistic regression test statistics
log_reg_test_data_statistics(logistic_model, log_X_test, log_y_test, "Logistic Regression") 

print("----------------------------------------------------------------------------", "\n")
#boosted tree statistics
log_reg_test_data_statistics(tree_model, tree_X_test, tree_y_test, "XGBoosted Tree") 


