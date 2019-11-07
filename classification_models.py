# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 07:23:25 2019

@author: Sravanii
"""

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

######################## Decision Tree Classiier ###########################
from sklearn.tree import DecisionTreeClassifier
dec_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dec_classifier.fit(X_train, y_train)

#############################################################################

######################## Random Forest Classifier #########################
from sklearn.ensemble import RandomForestClassifier
rand_classifier=RandomForestClassifier()
rand_classifier.fit(X_train, y_train)

###########################################################################

######################## Xg Boost Classifier ###############################
from xgboost import XGBClassifier
xg_classifier = XGBClassifier(n_estimators=100, max_depth=6, silent=False)
xg_classifier.fit(X_train, y_train)

###########################################################################

###################### SVM ################################################
from sklearn.svm import SVC
svm_classifier =SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

############################################################################

##################### Logistic Regression ##################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

############################################################################

########## Predicting the Test set results #################################
y_pred_dec = dec_classifier.predict(X_test)
y_pred_rand=rand_classifier.predict(X_test)
y_pred_xg=xg_classifier.predict(X_test)
y_pred_svm=svm_classifier.predict(X_test)
y_pred_logreg=logreg.predict(X_test)

############################################################################

################ Making the Confusion Matrix ###############################
from sklearn.metrics import confusion_matrix
dec_cm = confusion_matrix(y_test, y_pred_dec)
rand_cm = confusion_matrix(y_test, y_pred_rand)
xg_cm = confusion_matrix(y_test, y_pred_xg)
svm_cm = confusion_matrix(y_test, y_pred_svm)
logreg_cm = confusion_matrix(y_test, y_pred_logreg)

############################################################################

############## Accuracy (tp + tn) / (p + n)#################################
from sklearn.metrics import accuracy_score
dec_accuracy=accuracy_score(y_test,y_pred_dec)
rand_accuracy=accuracy_score(y_test,y_pred_rand)
xg_accuracy=accuracy_score(y_test,y_pred_xg)
svm_accuracy=accuracy_score(y_test,y_pred_svm)
logreg_accuracy=accuracy_score(y_test,y_pred_logreg)

accuracy=[dec_accuracy,rand_accuracy,xg_accuracy,svm_accuracy,logreg_accuracy]
a1=pd.DataFrame(accuracy)
############################################################################

############### Precision  tp / (tp + fp)###################################
from sklearn.metrics import precision_score
dec_precision = precision_score(y_test, y_pred_dec)
rand_precision = precision_score(y_test, y_pred_rand)
xg_precision = precision_score(y_test, y_pred_xg)
svm_precision = precision_score(y_test, y_pred_svm)
logreg_precision = precision_score(y_test, y_pred_logreg)

precision=[dec_precision,rand_precision,xg_precision,svm_precision,logreg_precision]
a2=pd.DataFrame(precision)
############################################################################

################ Recall: tp / (tp + fn) ###################################
from sklearn.metrics import recall_score
dec_recall = recall_score(y_test,y_pred_dec)
rand_recall = recall_score(y_test,y_pred_rand)
xg_recall = recall_score(y_test,y_pred_xg)
svm_recall = recall_score(y_test,y_pred_svm)
logreg_recall = recall_score(y_test,y_pred_logreg)

recall=[dec_recall,rand_recall,xg_recall,svm_recall,logreg_recall]
a3=pd.DataFrame(recall)
############################################################################

################# F1: 2 *(precision*recall)/ (precision+recall) ############
from sklearn.metrics import f1_score
dec_f1 = f1_score(y_test,y_pred_dec)
rand_f1 = f1_score(y_test,y_pred_rand)
xg_f1 = f1_score(y_test,y_pred_xg)
svm_f1 = f1_score(y_test,y_pred_svm)
logreg_f1 = f1_score(y_test,y_pred_logreg)

f1_score=[dec_f1,rand_f1,xg_f1,svm_f1,logreg_f1]
a4=pd.DataFrame(f1_score)
#############################################################################

################# ROC AUC####################################################
from sklearn.metrics import roc_auc_score
dec_roc_auc = roc_auc_score(y_test,y_pred_dec)
rand_roc_auc = roc_auc_score(y_test,y_pred_rand)
xg_roc_auc = roc_auc_score(y_test,y_pred_xg)
svm_roc_auc = roc_auc_score(y_test,y_pred_svm)
logreg_roc_auc = roc_auc_score(y_test,y_pred_logreg)

roc_auc=[dec_roc_auc,rand_roc_auc,xg_roc_auc,svm_roc_auc,logreg_roc_auc]
a5=pd.DataFrame(roc_auc)
#############################################################################

model=['Decision Tree','Random Forest','XG Boost','SVM','Logistic Regression']
aa=pd.DataFrame(model)
Result=pd.concat([aa,a1,a2,a3,a4,a5],axis=1)
Result.columns=['MODEL','Accuracy','Precision','Recall','F1_Score','ROC_AUC']
Result.to_csv("Social_Networks.csv",index=False)