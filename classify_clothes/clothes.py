# -*- coding: utf-8 -*-
"""Created on Sat Feb 22 09:38:10 2020 @author: Nour"""

""" it is the code to classify 10 types of clothes using ANN 
the data is obtain from kaggle
the accuracy of model is 88% """ 


#import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist #clothes_data
import tensorflow as tf

#%% Data Preprocessing
'''Data Preprocessing'''

### 1-load data images ###
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

### 2-normalize images ###
#normalize each pixel [0, 1] instead[0,255] to train ANN model faster 
x_train=x_train/255.0
x_test=x_test/255.0

### 3-Reshape of the dataset ###
#in the data rows are images , and columns are pixels (28*28)
#must change from 2d to 1d (length*width : 28*28=784) so row 6000,col 784
x_train=x_train.reshape(-1,28*28) # -1 all rows
x_test=x_test.reshape(-1,28*28) 
x_train.shape  #to ensure and know the size after reshape

#%% Building ANN
"""Building ANN"""

### 1- define model
model=tf.keras.models.Sequential()

### 2- adding first layer 
model.add(tf.keras.layers.Dense(units=128,activation='relu',input_shape=(784, ))) 
#model.add(tf.keras.layers.Dense(units=128,activation='relu')) #adding 2nd layer

### 3- add dropout layer
model.add(tf.keras.layers.Dropout(0.2)) 
#20% of neurons =0 deactivated ,not be updated by backpropagation so less chance for overfitting ,enhance training 

### 4- add output layer
model.add(tf.keras.layers.Dense(units=10 ,activation='softmax')) #  (no.of outputs=10)

### 5- compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
#adam optimizer ==> for stochastic gradient & loss:for error & metrics:for categorical accuracy

model.summary()

### 6- fit model
model.fit(x_train,y_train,epochs=10) #epochs: no. of train backpropagation

### 7- evaluate model
test_loss,test_accuraacy=model.evaluate(x_test,y_test)
print("test accuracy: {}".format(test_accuraacy))

### 8- saving model
#using json
model_json=model.to_json() 
with open("fashion model.json","w") as json_file:
    json_file.write(model_json)
    
model.save_weights("fashion model.h5")   #save weights
print("Saved model to disk")


