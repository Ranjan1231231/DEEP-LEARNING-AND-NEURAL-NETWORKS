#CONVOLUTIONAL NEURAL NETWORKS
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D #to add convolutional layer
from keras.layers.convolutional import MaxPooling2D #add pooling layer
from keras.layers import Flatten #to flatten data for fully connected layers
#IMPORTING THE KERAS DEFAULT DATASET
from keras.datasets import mnist
#LOADING THE DATA
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
#normalizing the pixel values to be between 0 and 1
X_train=X_train/255 #normalize training data
X_test=X_test/255#normalize the test data
#converting the target values to binary categories
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]




#defining the function that creates our model , starting with one set of convolutional and pooling layers
#Convolutional layer with one set of convolutional and pooling layer



# def convolutional_model():
#     model= Sequential()
#     model.add(Conv2D(16,(5,5),strides=(1, 1),activation='relu',input_shape=(28,28,1)))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100,activation='relu'))
#     model.add(Dense(num_classes,activation='softmax'))
#     model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#     return model
#
# #calling hte function to creating the model and then training and evaluating it
# #building the model
# model=convolutional_model()
# #fit the model
# model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=2)#verbose is a moethod to show the detailed information
# scores=model.evaluate(X_test,y_test,verbose=0)
# print("Accuracy:{}\n Error:{}".format(scores[1],100-scores[1]*100))

#Convolutional Layer with two sets of convolutional and pooling layers
def convolutional_model():
    model=Sequential()
    model.add(Conv2D(16,(5,5),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
#building the model
model=convolutional_model()
#fit the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=2)
#evalute the model
scores=model.evaluate(X_test,y_test,verbose=0)
print('Accuracy:{}\nError:{}'.format(scores[1],100-scores[1]*100))
