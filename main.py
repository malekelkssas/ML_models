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

dftrain = pd.read_csv(r'C:\Users\nice\Desktop\ai\training data\train.csv')  #training data
dfeval = pd.read_csv(r'C:\Users\nice\Desktop\ai\test data\eval.csv')        #testing data
y_train = dftrain.pop('survived')   #we remove the "survived" column and save it
y_eval = dfeval.pop('survived')     #we remove the "survived" column and save it


# categorical_columns = ['sex', 'n_siblings_spouses', 'parch','class', 'deck', 'embark_town', 'alone']
# numerical_columns = ['age', 'fare']
categorical_columns = []
numerical_columns = []
for i in dftrain:
    if(isinstance(dftrain[i][0],str)):
        categorical_columns.append(i)
    else:
        numerical_columns.append(i)
feature_columns = []
for feature_name in categorical_columns:
    vocabulary = dftrain[feature_name].unique()     #give a list of all unique values in this column i.e: sex ->['male','female']
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))      #this feature column needed for linear regression --> it's like make a specific number or id for each category
    # print(vocabulary)
    # break
for feature_name in numerical_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
#the following code is to broke our data into epochs and batches
#to deal with data (pandas data) we need to convert our data into (tf.data.Dataset) object as model need this data to be able to work --> this can be done by input function
#print(feature_columns)
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):      #(data frame(pandas),y_train or y_eval)
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  #create (tf.data.Dataset) object and it's label
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)      #split into batches and epoch it
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)  #i don't need to do that in test data i just turn it into (tf.data.Dataset) object

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) #that make the linear model for us , we pass the feature_columns in line 22

# # now training the model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! we just pass the prev functions :')

linear_est.train(train_input_fn) # that means train the data , the data  ((tf.data.Dataset) object) from line 41
result = linear_est.evaluate(eval_input_fn) # that means test the data,get the model metrics/stats by testing on testing data and save it in result variable
#
clear_output() # clear the output
print("accuracy = "+str(result['accuracy']*100)+"%")
#print(result)
result = list(linear_est.predict(eval_input_fn)) # the value of each prediction ,I most care about the probability
# print(result)
# print(result[0]) #'probabilities': array([0.84687304, 0.15312691] that means he will not survive by about 84% and will survive by 15% as we assume that survived = 0 means not survive
# print(result[0]['probabilities'])
# print(result[0]['probabilities'][0]) #not survived percentage
# #the full information about the person
# print(dfeval.loc[0])
def check(value):
    if(value):
        return "survived"
    else:
        return"not survived"
for i in range (len(dfeval)):
    print(dfeval.loc[i])
    print("most probably he/she survived by "+str(result[i]['probabilities'][1]*100)+"%")
    print("actually he/she is "+check(y_eval.loc[i]))

