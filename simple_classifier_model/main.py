from  __future__  import absolute_import,division,print_function,unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
import keras
#tensorflow.compat.v2.feature_column
def fc():
    return tf.compat.v2.feature_column


#1-organizing data
#__________________
csv_column_names = ['sepalLength','sepalWidth','petalLength','petalWidth','species']
species = ['Setosa','Versicolor','Virginica']

train = pd.read_csv(r'training\iris_training.csv',names=csv_column_names,header=0)  #training data (we do it this way because the data is stored by different way),header = 0 ->means the first row(which will ignore because I give it the columns name)
test = pd.read_csv(r'testing\iris_test.csv',names=csv_column_names,header=0)        #testing data
y_train = train.pop('species')   #we remove the "species"because the training about know the species of the flowers
y_test = test.pop('species')

feature_columns = []
for key in train:
    feature_columns.append(tf.feature_column.numeric_column(key=key))

#2-make input function
#_______________________
# it will be different as instead of make neasted functions that return the inner function, I will the outer function in the term of lamda expression

def input_function(data,label,training = True,batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices((dict(data), label))  #create (tf.data.Dataset) object and it's label
    if training:
        ds = ds.shuffle(1000).repeat()
    return ds.batch(batch_size)



#3-build the model
#build a DNN with 2 hidden layers with 30 in first and 10 in second hidden nodes each

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30,10],   # each hidden layers nodes
    n_classes=3 # because model must choose between three flowers
)

#4-train the model and test it


classifier.train(
    input_fn= lambda : input_function(train,y_train),
    steps=5000          #steps here like epochs
)

eval_result = classifier.evaluate(input_fn= lambda : input_function(test,y_test,training=False))

#5-print the accuracy

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))      #the accuracy shall be print in this format

print('Test set accuracy = '+str(eval_result['accuracy']*100)+'%')
#6- prediction
#here I will take input from user and predict the flower
def input_fn(data,batch_size=256):        #this func is making to use in predition (because it is a must to deal with  (tf.data.Dataset) object),and i don't want label here
    return tf.data.Dataset.from_tensor_slices(dict(data)).batch(batch_size)

features = ['sepalLength','sepalWidth','petalLength','petalWidth']
predict = {}

for feature in features:
    valid = True
    while valid:
        val = input(feature +": ")
        if not val.isdigit() : valid=False
    predict[feature] = [float(val)]         #this is how must be the format

prediction = classifier.predict(input_fn= lambda: input_fn(predict)) #The predict method returns a Python iterable, yielding a dictionary of prediction results for each example.
for pred_dict in prediction:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]        #probabilities = [x1,x2,x3] , there is a probability for each kind and the model choose the higher
    print('Prediction is "{}" ({:.1f}%)'.format(species[class_id], 100 * probability))

