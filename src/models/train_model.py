import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data/')
sys.path.append("../src")
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

df_train = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/df feat eng done.csv")

X, y = md.x_y_split(df_train, -1)

#80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)


#will functionalize the below code 

# fit initial model
logistic = LogisticRegression(max_iter = 10000, random_state = 21)
logistic.fit(X_train, y_train)

# predict on the training data
log_pred_train = logistic.predict(X_train)
log_pred_test = logistic.predict(X_test)

# display the inital model scores for training data
accuracy_training = logistic.score(X_train, y_train)
accuracy_testing = logistic.score(X_test, y_test)

# display confusion matrix for training data
print("Logistic Regression Model Training Data Confusion Matrix :")
print(confusion_matrix(y_train, log_pred_train)) 
print('The accuracy score for the training data is:', accuracy_training.round(3), "\n")


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
df_coef

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



'''
#running LazyPredict regressor to get a sense of what other models should I look at
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models_reg,predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models_reg)

#running LazyPredict classifier to get a sense of what other models should I look at
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models_clf,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models_clf)'''