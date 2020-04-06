#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

count = CountVectorizer()
file_train = 'training.csv'
file_test = 'test.csv'
df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
train_text = df_train['article_words']
test_text = df_test['article_words']
all_test = pd.concat([train_text,test_text])

bag_of_words = count.fit(all_test)
X_train = bag_of_words.transform(train_text)
y_train = df_train['topic'].to_list()
X_test = bag_of_words.transform(test_text)
y_test = df_test['topic'].to_list()

y = y_test + y_train
topic_dict = dict.fromkeys(y)
topic_dict.update((k,i) for i,k in enumerate(topic_dict))
y_train = [topic_dict[k] for k in y_train]
y_test = [topic_dict[k] for k in y_test]


# In[ ]:




