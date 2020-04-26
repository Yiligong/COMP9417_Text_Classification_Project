#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

training_file = pd.read_csv('training.csv')
test_file = pd.read_csv('test.csv')

#One Hot Encoding to quantitatively represent the topics
encoding = {'topic' : {'IRRELEVANT' : 0, 
                       'ARTS CULTURE ENTERTAINMENT':1, 
                       'BIOGRAPHIES PERSONALITIES PEOPLE':2, 
                       'DEFENCE' : 3, 
                       'DOMESTIC MARKETS' : 4, 
                       'FOREX MARKETS' : 5, 
                       'HEALTH' : 6, 
                       'MONEY MARKETS' : 7,
                       'SCIENCE AND TECHNOLOGY' : 8, 
                       'SHARE LISTINGS' : 9, 
                       'SPORTS' :10}}

#Replacing topics with the relevant numbers for training data
training_file = training_file.replace(encoding)

test_file = test_file.replace(encoding)
#print(test_file)

#Training and Testing split - X and Y
x_train = training_file['article_words']
y_train = training_file['topic'].tolist()
x_test = test_file['article_words']
y_test = test_file['topic'].tolist()

#Transforming the testing and training and create bag of words
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
#print(x_train)
#print(y_train)

hparams = {"alpha": [0.01, 0.1, 0.5, 1.0, 10, 100], "fit_prior":['True','False']}

from sklearn.model_selection import GridSearchCV
#GS_estimator = GridSearchCV(MultinomialNB(), hparams, cv=5, scoring="accuracy")
GS_estimator = GridSearchCV(MultinomialNB(), hparams, refit = True, verbose = 2)
clf = GS_estimator.fit(x_train, y_train)
print(GS_estimator.best_estimator_)

#implement of MultinomialNB
mnb = MultinomialNB(clf.best_estimator_.alpha)

model = mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)

train_accuracy_score = accuracy_score(y_train,model.predict(x_train))
test_accuracy_score = accuracy_score(y_test,model.predict(x_test))
print(f'Accuracy Score for training data : {train_accuracy_score}.\n')
print(f'Accuracy Score for testing data : {test_accuracy_score}.\n')

#Report
print(classification_report(y_test, y_predict))
#ds = pd.read_csv("/home/nikita/Downloads/sample-data.csv")
from sklearn.metrics.pairwise import cosine_similarity
def recommendation(test,y_predict,train_data,topic_dict):
    test_data = test.copy(deep=True)
    test_data['topic'] = y_predict
    topic_list = [i for i in range(11)]
    tf = TfidfVectorizer()
    model = tf.fit(train_data['article_words'])
 
    for t in topic_list:
        if not test_data[test_data['topic']==t].empty:
            
            tfidf_train = model.transform(train_data[train_data['topic']==t]['article_words']).toarray()
            tfidf_test = model.transform(test_data[test_data['topic']==t]['article_words']).toarray()

            test_article_number = test_data[test_data['topic']==t]['article_number'].tolist()
           
            cosine = cosine_similarity(tfidf_test,tfidf_train)
            cosine = np.sort(cosine)
            suggested_list = np.argsort(cosine[:,-1])[-10:].tolist()
            article_number = ",".join([str(test_article_number[i]) for i in suggested_list])         
            print(f"For topic {topic_dict[t]} recommending article {article_number}")
            
topic_dict = {encoding['topic'][k] : k for k in encoding['topic']}
recommendation(test_file,y_predict,training_file,topic_dict)
