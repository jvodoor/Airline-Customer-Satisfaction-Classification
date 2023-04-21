import sys
sys.path.insert(0, '/Users/jvodo/DATA%204950/DATA-4950-Capstone/src/data/')
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
import train_model as tm

def log_reg_predict(logistic, X_test, y_test):
    log_pred_test = logistic.predict(X_test)

    accuracy_testing = logistic.score(X_test, y_test)

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


    # predict on the test data X_test
    log_pred = logistic.predict(X_test)

    # returns the probability for both class labels
    logexport_prob = logistic.predict_proba(X_test) 

    #print confusion matrix for testing data
    print("Logistic Regression Model Testing Data Confusion Matrix :")
    print(confusion_matrix(y_test, log_pred_test)) 
    print('The accuracy score for the testing data is:', accuracy_testing.round(3))

df_test = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df feat eng done.csv")

X, y = md.x_y_split(df_test, -1)

#80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

logistic_model = tm.log_reg_train(X_train, X_test, y_train, y_test)

log_reg_predict(logistic_model, X_test, y_test) 