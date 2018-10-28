#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
#import sys
from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess("svm/word_data.pkl", "svm/email_authors.pkl")

#########################################################
### your code goes here ###

#########################################################


#To reduce the training set so the time taken to train is lower
#Tradeoff is accuracy

features_train = features_train[:len(features_train)//100] 
labels_train = labels_train[:len(labels_train)//100] 

from sklearn.svm import SVC

#clf = SVC(kernel="linear")

clf = SVC(kernel='rbf', C=10000.0, gamma='scale')

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
prediction_results = clf.predict(features_test)
print("Predicting time:", round(time()-t1, 3), "s")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, prediction_results)
print("accuracy : ", accuracy)

#Label for the 10th test data 

label9 = prediction_results[10]
label26 = prediction_results[26]
label50 = prediction_results[50]

print("10th 26th & 50th predictions: ", label9, label26, label50)

# Number of emails predicted to be Chris's

from collections import Counter
print(Counter(prediction_results))

