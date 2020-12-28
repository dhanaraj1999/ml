# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:11:15 2020

@author: asus
"""

#lab6
#Document Classification using Naive Bayes Classifier
import pandas as pd
txt=pd.read_csv('Text.csv',names=['text','label']) #Tabular form data
print(txt)
print('\nTotal instances in the dataset is: ',txt.shape[0])

txt['labelnum'] = txt.label.map( {'pos':1, 'neg':0} )
X = txt.text
Y = txt.labelnum

# Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=0)
print('\nDataset is split into Training and Testing samples')
print('Total training instances :', xtrain.shape[0])
print(xtrain)
print('Total testing instances :', xtest.shape[0])
print(xtest)

# Output of count vectoriser is a sparse matrix
# CountVectorizer - stands for 'feature extraction'
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain) #Sparse matrix
xtest_dtm = count_vect.transform(xtest)
print('\nTotal features extracted using CountVectorizer:', xtrain_dtm.shape[1])

print('\nFeatures for training instances are:')
df = pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df.columns)

print('\nDocument term matrix is:\n ')
print(df)

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

print('\n---Classstification results of testing samples are given below----')
for doc, p in zip(xtest, predicted):
    pred = 'pos' if p==1 else 'neg'
print('%s -> %s ' % (doc, pred))

#printing accuracy metrics
from sklearn import metrics
print('\nAccuracy of the classifer is: ', metrics.accuracy_score(ytest, predicted))
print('Recall of the classifer is: ', metrics.recall_score(ytest, predicted))
print('Precison of the classifer is: ', metrics.precision_score(ytest, predicted))
print('Confusion matrix is: ')
print(metrics.confusion_matrix(ytest, predicted))