import sys
import path
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']
import make_dataset as md
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_tree
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
import statsmodels as sm
import matplotlib.pyplot as plt

from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor




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


def XG_Boost_Train(X_train, y_train):
    xgbc = XGBClassifier()

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
        max_delta_step=0, max_depth=3, min_child_weight=1, max_leaves = 50, missing=None,
        n_estimators=10, n_jobs=1, nthread=None,
        objective='multi:softprob', random_state=21, reg_alpha=0,
        reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
        subsample=1, verbosity=1) 
    xg_model = xgbc.fit(X_train, y_train)
    scores = cross_val_score(xg_model, X_train, y_train, cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())

    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(xg_model, X_train, y_train, cv=kfold )
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    # predict on the training data
    xg_pred_train = xg_model.predict(X_train)

    # display the inital model scores for training data
    accuracy_training = xg_model.score(X_train, y_train)

    # display confusion matrix for training data
    print("XG Boost Model Training Data Confusion Matrix :")
    print(confusion_matrix(y_train, xg_pred_train), "\n") 
    print('The accuracy score for the training data is:', accuracy_training.round(3), "\n")


    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(xg_model, num_trees=0, ax=ax)
    plt.show()
    return xg_model


 



#split dataset into X df featuring every column but target, and y which is just the target
X, y = md.x_y_split(df_train, -1)



#df 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)


#logistic_model = log_reg_train(X_train, X_test, y_train, y_test)
#df_coef = extract_coefs(logistic_model)

xg_model = XG_Boost_Train(X_train, y_train)

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
