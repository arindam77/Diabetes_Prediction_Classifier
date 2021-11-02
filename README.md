# Diabetes_Prediction_Classifier


The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.The dataset used for this project was the pima-diabetes dataset. Further context of the problem statement can be found on https://www.kaggle.com/uciml/pima-indians-diabetes-database


Tested with 4 different models and was able to achieve a maximum accuracy of 82 percent on the test data.

The different models used in this project were:

1.Random Forest

2.Support Vector Classifier

3.XGBoost

4.LightGBM


Tested with  5 fold cross validation and 10 fold cross validation and different hyperparameter optimization techniques for performance improvement.


The highest accuracy on the test data with 5 fold cross validation was obtained with Support Vector Classifier. The accuracy was 78%.

The highest accuracy on the test data with 10 fold cross validation was obtained with Random Forest Classifier. The accuracy was 82%.

Observation: The performance of all the other models significantly improved while using 10 fold cross validation.


So, the best model for the project was identified as Random Forest. The classification report for the model can be found below:

Accuracy score of the training data with Random Forest:  0.9315960912052117

Accuracy score of the test data with Random Forest:  0.8246753246753247

              precision    recall  f1-score   support

           0       0.87      0.88      0.87       107
           1       0.72      0.70      0.71        47

    accuracy                           0.82       154
   macro avg       0.79      0.79      0.79       154
weighted avg       0.82      0.82      0.82       154




