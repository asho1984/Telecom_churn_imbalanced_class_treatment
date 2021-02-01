# Telecom_churn_imbalanced_class_treatment

# How to train and validate classification model with imbalanced data.

## Introduction

Classification problems often have imbalanced classes. Examples include identifying credit card/loan frauds, fake news and people leaving a service provider to name a few. In such cases only a fraction of the cases are actually positive. In this model we are trying to identify and predict customers who are at high risk of churn. Since, there are only two classes i.e. churn and not churn, it is a binary classification problem. The dataset shows that the customers who churn form only about 1% of the population, hence the classes are highly imbalanced.
In this work two models using ensemble techniques i.e. XG Boost and Random Forest, to get higher performance. Another model was built using RFE logistic regression followed by Lasso regularization for feature selection. 

## Evaluation Metrics and Training data balancing

There are primarily two aspects of heavily imbalanced datasets that need special treatment i.e. the right choice of evaluation metrics and balancing the representation of various classes. Having 99% of the data as negative the model might have 99% accuracy by identifying all the cases as negative, but won’t be of any use since it can’t identify any True positives. In such cases one of the several data balancing techniques i.e. oversampling minority class, under sampling majority class etc. (https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/) need to be utilized to get a more equal representation of all the classes. Also, depending upon the objective of the problem statement, relative importance of True and False predictions and type of output required (class label or probability) appropriate evaluation metrics need to be utilized (https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/). In this work “Synthetic Minority Oversampling Technique” (SMOTE) has been utilized to balance the two classes in the training data and f2 score has been used as the evaluation metrics. 

## Right way of Oversampling and model validation

This leads to another peculiar aspect of modelling with imbalanced classes i.e. training and validation. If both the training and validation data are artificially balanced, the model won’t be a true representation of the actual data and hence will fail when tested on unseen or out of time data. There are several strategies that could be adopted to address this issue and ensure that the developed model performs equally well on unseen data, some of them involve training the model on artificially balanced data and correcting the probabilities thus calculated to suit the actual conditions (https://www.analyticsvidhya.com/blog/2014/01/logistic-regression-rare-event/),(https://www.knime.com/blog/correcting-predicted-class-probabilities-in-imbalanced-datasets).  
In this work the model is trained and cross validated for hyper parameter tuning, but only the training data is oversampled and model is validated on imbalanced data (https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html). For hyper parameter tuning using GridSearchCV with Kfold cross validation and simultaneously oversampling the training folds through SMOTE, the pipeline method is used. The model thus obtained was found to work equally well on test data.

## Model 1. XGboost Classifier

XGboost (extreme gradient boosting) is a very powerful algorithm and is an advanced application of gradient boosting algorithm (GBM). In GBM the model is sequentially improved upon by giving lesser weightage to data points which were correctly predicted by previous modelsreby improving the model by better predicting the remaining data points (https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/). In addition to the benefits associated with GBM, XGboost provides features like regularization, parallel processing, flexibility with respect to evaluation criteria etc. to name a few. However, this also makes hyper parameter tuning an essential part of developing a better model (https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). 

## Model 2. RandomForest Classifier

Random Forest Classifier (RFC) uses the bagging ensemble technique wherein the average outcome of several independent classifiers (decision tree) is considered to arrive at the final classification. RFC is also a very powerful technique and can be utilized successfully to solve very complex problems. The most important aspect of these classifiers is that it uses an ensemble of trees which are trained on a subset of the data and features and to some extent immune to overfitting. Like XGboost hyper parameter plays very important role in obtaining a better performing model (https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/). 

## Model 3. RFE + Lasso regularization

To identify the most important features affecting customer churn a Logistic regression model is built using RFE followed by Lasso regularization technique for feature selection. Hyper parameter tuning with Kfold cross validation was carried out to find the optimum value of regularization term alpha. Since the predict function for logistic regression models only provided the probabilities, classification evaluation metrics like recall score, f1 score etc. can’t be used. Area Under the Curve (AUC) for precision recall curve provides a good alternative evaluation metrics to develop a model to give higher recall score.

## Results 

### XGboost: 
		Precision    Recall  f1-score   support

           0       0.98      0.86      0.92      8251
           1       0.36      0.85      0.51       750

    accuracy                           0.86      9001
   macro avg       0.67      0.86      0.71      9001
weighted avg       0.93      0.86      0.89      9001

### Random forest:
           	Precision    Recall  f1-score   support

           0       0.98      0.79      0.87      8251
           1       0.27      0.85      0.41       750

    accuracy                           0.79      9001
   macro avg       0.63      0.82      0.64      9001
weighted avg       0.92      0.79      0.84      9001

### RFE + Lasso Regularization:
		Precision    Recall  f1-score   support

         0.0       0.98      0.77      0.86      8251
         1.0       0.24      0.81      0.37       750

    accuracy                           0.77      9001
   macro avg       0.61      0.79      0.62      9001
weighted avg       0.92      0.77      0.82      9001

![Most important features](https://user-images.githubusercontent.com/62643813/106435276-ae7a7680-6498-11eb-80d5-85eebceb897a.PNG)

Figure 1 Most important features

## Conclusion:

-Using SMOTE for balancing the data classes by oversampling is an effective method to develop classification model with imbalanced data.
-Using artificially balanced data for training but actual data for validation ensured that the model performed equally well on unseen data. The pipeline method has simplified this process by integrating it with GridSearchCV for hyper parameter tuning.
-The ensemble techniques yield better results than traditional classification methods.
-Hyper parameter tuning plays a very important role in improving the performance of ensemble based models.

Since the objective of developing this prediction model is to identify as many customers as possible who could probably "Churn" and leave the network, the Recall_score of the "Churn" cases is considered for selecting the model. A higher recall score means that the model is able identify a higher % of customers who could possibly "Churn".
The Recall_Score for both XGboost and RandomForest models is same i.e 85%, but XGboost has a better Precision i.e. 36% compared to 27% of Randomforest model, corresponding to the same Recall_score. Hence, the cost of deployment of XGboost model is less and is the proposed model for deployment.
However, both models should be tested on out of time data and the decision for final deployment should be based on its outcome.

