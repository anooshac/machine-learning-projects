#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets.base import load_iris
from sklearn.metrics import accuracy_score

#load iris dataset
iris=load_iris()

#get data, column names, labels, etc. as list
data=iris["data"]
feature_names=iris["feature_names"]
labels=iris["target"] 

#create dataframe with data and insert the labels
df=pd.DataFrame(data, columns=feature_names)
output=pd.DataFrame(labels, columns=['target'])

#default=5
model=KNeighborsClassifier(n_neighbors=6)

#train model
#if getting this error, it is because a matrix with 1 column
#is being passed in when a 1d array is expected. ravel() will work.
#DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel(). if name == 'main':
model.fit(df, output.values.ravel())
model.score(df, output.values.ravel())

#predict output
predicted=model.predict(df)
print predicted
print "Accuracy: ", accuracy_score(output, predicted)
