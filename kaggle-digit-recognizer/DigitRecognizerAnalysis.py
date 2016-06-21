import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
import urllib
from sklearn import preprocessing
import matplotlib.pyplot as plt


#Load Expedia dataset 
train=pd.DataFrame.from_csv("train-3.csv")
print train.shape
train=train.reset_index()
#print train.head(100)

test=pd.DataFrame.from_csv("test.csv")
print test.shape
test=test.reset_index()
#print test.head(100)

#Get unique pixel values in train and test
unique_train_pixels=pd.unique(train.ix[:, 'pixel0':'pixel783'].values.ravel())
#print "Unique Train Values:", unique_train_pixels
unique_test_pixels=pd.unique(test.values.ravel())
#print "Unique Test Values:", unique_test_pixels

#Check if pixels in test are a subset of pixels in train. If yes, easier to do predictions
#print "Test values in Train?", np.in1d(unique_test_pixels, unique_train_pixels, assume_unique=True)

#Check if there is linear correlation between pixel<x> columns and label
#If yes, we should dive into the columns with correlation. Linear / logistic regression may work well with the data.
#In this case, makes sense that there is no correlation - higher pixel values does not mean that label value will be higher
#print "Correlation:", train.corr()["label"]

#Check that the algorithm used gives good accuracy by using part of the training set to validate
train_train, train_test=train_test_split(train, test_size=0.3)

#Train model
model=RandomForestClassifier(n_estimators = 100, oob_score = True, random_state =10, max_features = "auto", min_samples_leaf = 20)
#model=KNeighborsClassifier(n_neighbors=6)


#if getting this error, it is because a matrix with 1 column
#is being passed in when a 1d array is expected. ravel() will work.
#DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel(). if name == 'main':
#To resolve this error, convert label values to int or str as float is not a valid label-type
#raise ValueError("Unknown label type: %r" % y) ValueError: Unknown label type: array
#model.fit(train_train.ix[:,'pixel0':'pixel783'], np.asarray(train_train.ix[:,'label'].astype(int)))
#print "model.score:", model.score(train_test.ix[:,'pixel0':'pixel783'], np.asarray(train_test.ix[:,'label'].astype(int)))
#print "cross validation score:", cross_validation.cross_val_score(model, train_train.ix[:,'pixel0':'pixel783'], train_train.ix[:,'label'], cv=3)
model.fit(train_train.ix[:,'pixel0':'pixel783'], train_train.ix[:,'label'].values.ravel())
print "model.score", model.score(train_test.ix[:,'pixel0':'pixel783'], train_test.ix[:,'label'].values.ravel())


#Predict output
#predicted=model.predict(train_test.ix[:,'pixel0':'pixel783'])
#print predicted
#print "Accuracy: ", accuracy_score(train_test.ix[:,'label'].astype(int), predicted)
