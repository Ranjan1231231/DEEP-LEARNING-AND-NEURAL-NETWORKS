import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense #type of layer
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train,y_train),(X_test,y_test)=mnist.load_data()
# print(X_train.shape)   #(60000, 28, 28)
# i=0
# while i<=100:
#     plt.imshow(X_train[i])
#     plt.show()
#     i=i+1
#flatting the image into one dimentional vectors to input
num_pixels=X_train.shape[1]*X_train.shape[2]# find size of one-dimensional vector
# print(num_pixels)
X_train=X_train.reshape(X_train.shape[0],num_pixels).astype('float32') # flatten training images
X_test=X_test.reshape(X_test.shape[0],num_pixels).astype('float32')# flatten test images
#Since pixel values can range from 0 to 255, let's normalize the vectors to be between 0 and 1.
X_train=X_train/255
X_test=X_test/255
#we need to divide our target variable into categories.
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]
# print(num_classes)

#BUILDING A NEURAL NETWORK
def classification_model():
    model=Sequential()
    model.add(Dense(num_pixels,activation='relu',input_shape=(num_pixels,)))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    #compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
#TRAIN AND TEST NETWORK
#build the model
model=classification_model()
#fit the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,verbose=2)
#evaluate the model
scores=model.evaluate(X_test,y_test,verbose=0)
print('Accuracy: {}%\n Error:{}'.format(scores[1],1-scores[1]))
model.save('classification_model.h5') #saving the model

#TO LOAD THE MODEL
# from keras.models import load_model
# pretrained_model = load_model('classification_model.h5')