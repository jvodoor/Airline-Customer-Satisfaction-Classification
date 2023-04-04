import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data/')
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

df_train = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df feat eng done.csv")

def log_reg_train(X_train, X_test, y_train, y_test):
    # fit initial model
    logistic = LogisticRegression(max_iter = 10000, random_state = 21)
    logistic.fit(X_train, y_train)

    # predict on the training data
    log_pred_train = logistic.predict(X_train)

    # display the inital model scores for training data
    accuracy_training = logistic.score(X_train, y_train)

    # display confusion matrix for training data
    print("Logistic Regression Model Training Data Confusion Matrix :")
    print(confusion_matrix(y_train, log_pred_train), "\n") 
    print('The accuracy score for the training data is:', accuracy_training.round(3), "\n")
    return logistic

def extract_coefs(logistic):
    log_intercept = logistic.intercept_ 
    beta_0 = log_intercept

    # extract log reg coefs
    coef = logistic.coef_[0]
    coef = np.array(coef)
    df_coef = pd.DataFrame(coef)
    df_coef = df_coef.T # transpose to match column names

    # column names
    names = X.columns
    df_coef.columns = names
    df_coef = df_coef.T # transpose for better view

    # sort coefficients in ascending order
    df_coef = df_coef.rename(columns = {0:'logregCV_coeff'})
    df_coef = df_coef.sort_values('logregCV_coeff')
    df_coef = df_coef.reset_index()
    df_coef = df_coef.rename(columns = {'index':'Variable_Names', 'logreg_coeff':'logregCV_coeff'})
    print("Coefficient importance for Logistic Regression:", "\n")
    print(df_coef)
    print ("\n")
    return df_coef


X, y = md.x_y_split(df_train, -1)

#80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)


#will functionalize the below code 

logistic_model = log_reg_train(X_train, X_test, y_train, y_test)

df_coef = extract_coefs(logistic_model)


#Overall accuracy and ROC curve values are fairly strong. 88% accuracy and .94 ROC curve indicates we can be quite confident
#in the predictors we have that predict a customer's satisfaction. That being said, we should explore removing
#the variables that don't have as strong of a feature importance in order to have a more explainable model,
#and potentially a better scoring model. We also need to look into alternate models to use, likely a neural net.



'''
#running LazyPredict classifier to get a sense of what other models should I look at
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models_clf,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models_clf)
'''
