#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: arindam
"""


#Importing all libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import xgboost
import lightgbm as lgb




## EDA & Data preprocessing



#Importing the dataset

data = pd.read_csv("pima-data.csv")


#Checking the number of rows and Columns in the dataset

data.shape


#Checking the statistical measures of the dataset

data.describe()


#Printing the first 5 rows of the dataset

data.head(5)


#Checking if any null value is present

data.isnull().values.any()


#Getting correlations of each features in dataset

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))


#Plotting heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corr_val = data.corr()


#Changing Target column data from boolean to number

output_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(output_map)
diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])


#Checking for Class Imbalance

(diabetes_true_count,diabetes_false_count)


#Separating data and labels

feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']
X = data[feature_columns].values
y = data[predicted_class].values


#Feature Scaling

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


#Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)


#Checking number of 0 values per feature

print("total number of rows : {0}".format(len(data)))
print("number of rows missing num_preg: {0}".format(len(data.loc[data['num_preg'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing thickness: {0}".format(len(data.loc[data['thickness'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


#Performing Imputation

fill_values = SimpleImputer(missing_values=0, strategy='mean')

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)




## Applying RandomForest Classification Algorithm with RandomizedSearchCV(for Hyperparameter optimization)



#Parameters to be used for RandomizedSearchCV-

random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}

random_forest_model = RandomForestClassifier(random_state=10)
rf_search=RandomizedSearchCV(random_forest_model,param_distributions=random_grid,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)


#Training the RandomForest Classifier

rf_search.fit(X_train,y_train.ravel())



#Getting the optimal hyperparameters

rf_search.best_estimator_



#Initializing RandomForest Classifier with the optimal hyperparameters
  
random_forest_model = RandomForestClassifier(max_features='sqrt', min_samples_leaf=2,
                       min_samples_split=10, n_estimators=130, random_state=10)



#Training the model with the updated hyperparameters
  
random_forest_model.fit(X_train, y_train.ravel())



#Accuracy score on the training data

rf_train_prediction = random_forest_model.predict(X_train)
training_data_accuracy = metrics.accuracy_score(rf_train_prediction, y_train)
print('Accuracy score of the training data with Random Forest: ', training_data_accuracy)


#Accuracy score on the test data

X_test_prediction = random_forest_model.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data with Random Forest: ', test_data_accuracy)


#Classification report

print(classification_report(y_test, X_test_prediction))




## Applying Support Vector Classification Algorithm with RandomizedSearchCV(for Hyperparameter optimization)


#Parameters to be used for RandomizedSearchCV-

param_SVC = {'C': [1, 10, 100, 1000], 
          'gamma': [0.001, 0.0001], 
          'kernel': ['rbf', 'linear'],}

svc_classifier = svm.SVC()
svc_search=RandomizedSearchCV(svc_classifier,param_distributions=param_SVC,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)


#Training the Support Vector Classifier

svc_search.fit(X_train,y_train.ravel())


#Getting the optimal hyperparameters

svc_search.best_estimator_



#Initializing Support Vector Classifier with the optimal hyperparameters

svc_classifier = svm.SVC(C=10, gamma=0.001)



#Training the model with the updated hyperparameters

svc_classifier.fit(X_train, y_train.ravel())



#Accuracy score on the training data

X_train_prediction = svc_classifier.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data with SVC: ', training_data_accuracy)



#Accuracy score on the test data

X_test_prediction = svc_classifier.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data with SVC : ', test_data_accuracy)


#Classification report

print(classification_report(y_test, X_test_prediction))




## Applying XGBOOST Algorithm with RandomizedSearchCV(for Hyperparameter optimization)



#Parameters to be used for RandomizedSearchCV-

xgb_params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

xgboost_classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(xgboost_classifier,param_distributions=xgb_params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)


#Training the XGBOOST Classifier

random_search.fit(X_train,y_train.ravel())


#Getting the optimal hyperparameters

random_search.best_estimator_



#Initializing XGBOOST Classifier with the optimal hyperparameters

xgboost_classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=5, missing=None, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)



#Training the model with the updated hyperparameters

xgboost_classifier.fit(X_train, y_train.ravel())


#Accuracy score on the training data

X_train_prediction = xgboost_classifier.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data with XGBOOST: ', training_data_accuracy)


#Accuracy score on the test data

X_test_prediction = xgboost_classifier.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data with XGBOOST : ', test_data_accuracy)


#Classification report

print(classification_report(y_test, X_test_prediction))




## Applying LightGBM Algorithm with RandomizedSearchCV(for Hyperparameter optimization)


#Parameters to be used for RandomizedSearchCV-

lgb_params = {

        'bagging_fraction': (0.5, 0.8),
        'bagging_frequency': (5, 8),

        'feature_fraction': (0.5, 0.8),
        'max_depth': (10, 13),
        'min_data_in_leaf': (90, 120),
        'num_leaves': (1200, 1550)

}

lgbm_classifier = lgb.LGBMClassifier()
random_search=RandomizedSearchCV(lgbm_classifier,param_distributions=lgb_params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)


#Training the LightGBM Classifier

random_search.fit(X_train,y_train.ravel())


#Getting the optimal hyperparameters

random_search.best_estimator_


#Initializing LightGBM Classifier with the optimal hyperparameters

lgbm_classifier =lgb.LGBMClassifier(bagging_fraction=0.8, bagging_frequency=8, feature_fraction=0.5,
               max_depth=13, min_data_in_leaf=90, num_leaves=1550)



#Training the model with the updated hyperparameters

lgbm_classifier.fit(X_train, y_train.ravel())


#Accuracy score on the training data

X_train_prediction = xgboost_classifier.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data with LGBMClassifier: ', training_data_accuracy)


#Accuracy score on the test data

X_test_prediction = xgboost_classifier.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data with LGBMClassifier : ', test_data_accuracy)


#Classification report

print(classification_report(y_test, X_test_prediction))




##Predicting for a single instance


input_data = (5,166,72,19,175,25.8,0.587,51, 2.4)

#Changing the input_data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

#Reshaping the array as we are predicting for a single instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#Standardize the input data

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = svc_classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
  
else:
  print('The person is diabetic')
