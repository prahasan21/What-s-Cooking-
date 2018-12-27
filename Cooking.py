# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 00:14:33 2018

@author: 18123
"""

import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train = pd.read_json("train.json")
test = pd.read_json("test.json")


#Data Cleaning:


def data_clean(ingredient):
    ingred = ingredient
    ingred = re.sub('[^a-zA-Z]', ' ',str(ingred))
    ingred = ingred.lower()
    ingred = ingred.split()
    lemmatizer = WordNetLemmatizer()
    ingred = [lemmatizer.lemmatize(w) for w in ingred if not w in set(stopwords.words('english'))]
    return (' '.join(ingred))



#train["Cleaned_ingredients"] = train["ingredients"].apply(lambda x:data_clean(x))
    

train.to_csv("train.csv", sep=',', index = False)


#Train and test splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['Cleaned_ingredients'],train['cuisine'], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)



#Bag of words
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10)
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.fit_transform(X_test).toarray()
    




#Model 1 Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='l1', C=1000)
#Fit the train data
log_reg.fit(X_train_bow,  y_train)
y_train_pred = log_reg.predict(X_train_bow)
y_test_pred_logreg = log_reg.predict(X_test_bow)
log_reg_score = accuracy_score(y_test,y_test_pred_logreg)
print("Logististic Regression score", log_reg_score)
  

#Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
classifier_mulnb = MultinomialNB()
classifier_mulnb.fit(X_train_bow, y_train)
y_test_pred_nulnb = classifier_mulnb.predict(X_test_bow)
mulnb_score = accuracy_score(y_test,y_test_pred_nulnb)
print("Multinomial Naive Bayes score", mulnb_score)


#SVM

from sklearn.svm import LinearSVC
clf_svm = LinearSVC(random_state=0, tol=1e-5)
clf_svm.fit(X_train_bow,  y_train)
y_test_predicted_svm = clf_svm.predict(X_test_bow)
score_test_svm = accuracy_score(y_test, y_test_predicted_svm)
print("Linear SVM score",score_test_svm)


#KNN

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=13)
neigh.fit(X_train_bow,  y_train)
y_test_predicted = neigh.predict(X_test_bow)
score_test_knn = accuracy_score(y_test, y_test_predicted)
print("KNN score",score_test_knn)

















