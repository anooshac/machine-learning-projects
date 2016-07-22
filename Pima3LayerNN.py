import numpy as np
import urllib
import pandas as pd
from sklearn.cross_validation import train_test_split

#create a dataset
D = 8 # dimensionality
K = 2 # number of classes

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

X = train.ix[:,'times pregnant':'age (years)']
y = train.ix[:,'target']
X2 = test.ix[:,'times pregnant':'age (years)']
y2 = test.ix[:,'target']

# initialize parameters randomly
h = 20 # size of hidden layer
h2 = 10
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,h2)
b2 = np.zeros((1,h2))
W3 = 0.01 * np.random.randn(h2,K)
b3 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-3
reg = 1e-2 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(50000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  hidden_layer2 = np.maximum(0, np.dot(hidden_layer, W2) + b2) 
  scores = np.dot(hidden_layer2, W3) + b3
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W3 and b3
  dW3 = np.dot(hidden_layer2.T, dscores)
  db3 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer2
  dhidden2 = np.dot(dscores, W3.T)
  # backprop the ReLU non-linearity
  dhidden2[hidden_layer2 <= 0] = 0
  # backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dhidden2)
  db2 = np.sum(dhidden2, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dhidden2, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW3 += reg * W3
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2
  W3 += -step_size * dW3
  b3 += -step_size * db3

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X2, W) + b) # note, ReLU activation
hidden_layer2 = np.maximum(0, np.dot(hidden_layer, W2) + b2) 
scores = np.dot(hidden_layer2, W3) + b3

predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y2))