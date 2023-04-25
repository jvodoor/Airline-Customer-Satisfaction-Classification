import sys
import path
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']
import make_dataset as md
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_tree
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns

from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor


df_train = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df feat eng done.csv")


def log_or_boost_train(X_train, X_test, y_train, log_or_boost):
    if log_or_boost == 'log':
        # fit initial model
        model = LogisticRegression(max_iter = 10000, random_state = 21)
        model.fit(X_train, y_train)

    if log_or_boost == 'boost':
        xgbc = XGBClassifier(max_depth = 5 )
        model = xgbc.fit(X_train, y_train)

    return model 


def log_or_tree_scores(model, X_train, y_train, log_or_tree):
    
    if log_or_tree == 'log':

        # predict on the training data
        log_pred_train = model.predict(X_train)

        # display the inital model scores for training data
        accuracy_training = model.score(X_train, y_train)

        # display confusion matrix for training data
        print("Logistic Regression Model Training Data Confusion Matrix :")
        print(confusion_matrix(y_train, log_pred_train), "\n") 
        print('The accuracy score for the training data is:', accuracy_training.round(3), "\n")
        


    if log_or_tree == 'tree':

        # predict on the training data
        xg_pred_train = model.predict(X_train)

        # display the inital model scores for training data

        accuracy_training = model.score(X_train, y_train)
        # display confusion matrix for training data
        print("XG Boost Model Training Data Confusion Matrix :")
        print(confusion_matrix(y_train, xg_pred_train), "\n") 
        print('The accuracy score for the training data is:', accuracy_training.round(3), "\n")

        fig, ax = plt.subplots(figsize=(40, 10))
        plot_tree(model, ax=ax)
        plt.show()
        


def graph_coefs(model, X_data, log_or_tree):
    
    if log_or_tree == 'log':
        log_intercept = model.intercept_ 
        beta_0 = log_intercept

        # extract log reg coefs
        coef = model.coef_[0]
        coef = np.array(coef)
        df_coef = pd.DataFrame(coef)
        df_coef = df_coef.T # transpose to match column names

        # column names
        names = X_data.columns
        df_coef.columns = names
        df_coef = df_coef.T # transpose for better view

        # sort coefficients in ascending order
        df_coef = df_coef.rename(columns = {0:'logregCV_coeff'})
        df_coef = df_coef.sort_values('logregCV_coeff')
        df_coef = df_coef.reset_index()
        df_coef = df_coef.rename(columns = {'index':'Variable_Names', 'logreg_coeff':'logregCV_coeff'})
        
        
        # display barplot of log reg coefs
        plt.figure(figsize=(12, 8))
        plot = sns.barplot(x="Variable_Names",y="logregCV_coeff",data= df_coef, palette ="YlGnBu")
        plot.bar_label(plot.containers[0], fmt='%.2g')

        plt.xticks(rotation=80)
        plt.title('The Barplot of Coefficients of Logistic Regression Model')
        plt.xlabel('Variable Names')
        plt.ylabel('Value of Coefficient')
        plt.show()
        

    if log_or_tree == 'tree':
        feat_imp = pd.DataFrame(model.feature_importances_)
        names = pd.DataFrame(list(X_data.columns))
        df_feat_imp = pd.concat([feat_imp, names], axis = 1)
        df_feat_imp.columns = ['Importance', 'Features']
        df_feat_imp.sort_values('Importance', ascending = False)

        Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=names)
        Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='b')
        plt.xlabel('Variable Importance')
        plt.gca().legend_ = None
        plt.show()
    
    
    print ("\n")






#split dataset into X df featuring every column but target, and y which is just the target
X, y = md.x_y_split(df_train, -1)



#df 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

#logistic regression
logistic_model = log_or_boost_train(X_train, X_test, y_train, 'log')
log_or_tree_scores(logistic_model, X_train, y_train, 'log')
graph_coefs(logistic_model, X, 'log')

#Overall accuracy and ROC curve values are fairly strong. 88% accuracy and .94 ROC curve indicates we can be quite confident
#in the predictors we have that predict a customer's satisfaction. That being said, we should explore removing
#the variables that don't have as strong of a feature importance in order to have a more explainable model,
#and potentially a better scoring model. Our alternative approach will be using an XGBoostCLassifier Tree model.

#XGBoost Tree classifier
xg_model = log_or_boost_train(X_train, X_test, y_train, 'boost')
log_or_tree_scores(xg_model, X_train, y_train, 'tree')

graph_coefs(xg_model, X, 'tree')


#evaluating our XGBoost decision tree, we appear to have a very high training accuracy score of 96%. This is a
#significant improvement from our logistic regression model, with very little hyper parameter tuning necessary.
#The one significant tuning I did was to limit the depth of the tree. While this in practice slightly reduces 
#the accuracy by less than 1%, in return we have a much more interpretable and explainable tree, which makes
#implementation in a business world a lot easier. 

#Looking at a comparison of plots, it's interesting how the most important variables differ significantly
#from logistic regression to XGBoost. The online boarding process before a flight is most significant for
#the XGBoost by a fair margin, while the Type of Travel, that is business or personal, is much more important
#for logistic regression. Gender represents a fairly insignifcant importance on both models, but let's try an
#alternate model without gender to see if that makes a difference.

print("----------------------------------------------------------------")

df_no_gender = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df alt no gender.csv")

#split dataset into X df featuring every column but target, and y which is just the target
no_gender_X, no_gender_y = md.x_y_split(df_no_gender, -1)

#df 80/20 train test split
no_gender_X_train, no_gender_X_test, no_gender_y_train, no_gender_y_test = train_test_split(no_gender_X, no_gender_y, test_size = 0.2, random_state = 21)

#logistic regression without gender
no_gender_logistic = log_or_boost_train(no_gender_X_train, no_gender_X_test, no_gender_y_train, 'log')
log_or_tree_scores(no_gender_logistic, no_gender_X_train, no_gender_y_train, 'log')
graph_coefs(no_gender_logistic, no_gender_X, 'log')

#XGBoost Tree classifier without gender
no_gender_xg_model = log_or_boost_train(no_gender_X_train, no_gender_X_test, no_gender_y_train, 'boost')
log_or_tree_scores(no_gender_xg_model, no_gender_X_train, no_gender_y_train, 'tree')
graph_coefs(no_gender_xg_model, no_gender_X, 'tree')



#Overall, logistic regression values changed minimally, and XG_Boost model did not change at all with the removal
#of gender as a predictor. Even though this doesn't definitively prove that there is no inherent Gender bias,
#given how Gender as it is is not an important feature, I do not expect there to be further bias as a result
#of its inclusion or exclusion. To be safe, we can continue with the removal of it. 

 

#running LazyPredict classifier to get a sense of what other models should I look at
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#models_clf,predictions = clf.fit(X_train, X_test, y_train, y_test)
#print(models_clf)

