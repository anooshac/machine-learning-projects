import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
import urllib

#fix seed for reproducability
seed=7
np.random.seed(seed)

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

#create model
model=Sequential()
model.add(Dense(8,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

#compile and train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.asarray(train.ix[:,'times pregnant':'age (years)']), np.asarray(train.ix[:,'target']), nb_epoch=150,batch_size=10)

#predict and get model metrics
scores=model.evaluate(np.asarray(test.ix[:,'times pregnant':'age (years)']), np.asarray(test.ix[:,'target']))
print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))