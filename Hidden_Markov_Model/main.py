from  __future__  import absolute_import,division,print_function,unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow_probability as tfp
import tensorflow as tf
import keras
#tensorflow.compat.v2.feature_column
def fc():
    return tf.compat.v2.feature_column

# given the following information
#_________________________________
# 1-cold days are encoded by a 0 and hot days are encoded by a 1
# 2-the first day in our sequence has an 80% chance of being cold
# 3-a cold day has a 30% chance of being followed by a hot day
# 4-a hot day has a 20% chance of being followed by a cold day
# 5- on each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day
#___________________________

#1-organizing data
#__________________

tfd = tfp.distributions     # it's just a shortcut
initial_distribution = tfd.Categorical(probs=[0.8,0.2]) #refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.7,0.3],[0.2,0.8]]) #refer to point 3 and 3 above , [0] for cold day and [1] for hot one
observation_distribution = tfd.Normal(loc=[0.,15.],scale=[5.,10.])  #refer to point 5 above , the loc argument represents the mean and the scale is the standard deviation

#2-build the model
#_________________

model = model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

#3- prediction
#______________
mean = model.mean()     #calculate the probability and # shape [7], elements approach 9.0

with tf.compat.v1.Session() as sess:
    print(mean.numpy())             # then we have the temperature in different days
