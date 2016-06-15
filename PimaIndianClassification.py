import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
import urllib
from sklearn import preprocessing
import matplotlib.pyplot as plt


#load Pima Indian dataset
url="http://goo.gl/j0Rvxq"

#download the file
raw_data=urllib.urlopen(url)

#get data, add column names and index
feature_names=["times pregnant", "plasma glucose conc.", "distolic blood pressure (mm Hg)", "triceps skin fold thickness (mm)", "2-hour serum insulin (mu U/ml)", "body mass index (kg/m^2)", "diabetes pedigree function", "age (years)", "target"]
dataset=pd.DataFrame.from_csv(raw_data)
dataset=dataset.reset_index()
dataset.columns=feature_names

#split into train and test set
train, test=train_test_split(dataset, test_size=0.3)

#normalize data
df_scaled_train=pd.DataFrame(preprocessing.scale(train), columns=feature_names)
df_scaled_test=pd.DataFrame(preprocessing.scale(test), columns=feature_names)

model=RandomForestClassifier(n_estimators = 100, oob_score = True, random_state =10, max_features = "auto", min_samples_leaf = 20)

#train model
#if getting this error, it is because a matrix with 1 column
#is being passed in when a 1d array is expected. ravel() will work.
#DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel(). if name == 'main':
#To resolve this error, convert label values to int or str as float is not a valid label-type
#raise ValueError("Unknown label type: %r" % y) ValueError: Unknown label type: array
model.fit(df_scaled_train.ix[:,'times pregnant':'age (years)'], np.asarray(df_scaled_train.ix[:,'target'].astype(int)))
print "Accuracy:", model.score(df_scaled_test.ix[:,'times pregnant':'age (years)'], np.asarray(df_scaled_test.ix[:,'target'].astype(int)))

#predict output
predicted=model.predict(df_scaled_test.ix[:,'times pregnant':'age (years)'])
print predicted