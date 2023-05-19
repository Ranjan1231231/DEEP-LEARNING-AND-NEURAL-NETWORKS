import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense #type of layer
import gc
concrete_data=pd.read_csv("concrete_data.csv")
# print(concrete_data.head())
# print(concrete_data.shape)
# print(concrete_data.describe())
# checking for null datas
# print(concrete_data.isnull().sum())

#SPLITTING THE DATA INTO PREDICTORS AND TARGET
concrete_data_columns=concrete_data.columns
predictors=concrete_data[concrete_data_columns[concrete_data_columns!='Strength']]
target=concrete_data['Strength']
# print(predictors.head(),target.head())



#NORMALIZING THE DATA by substracting the mean and dividing by the standard deviation.
predictors_norm=(predictors-predictors.mean())/predictors.std()
# print(predictors_norm.head())
n_cols=predictors_norm.shape[1]#number  of predictors that will need when building our network
#BUILDING A NEURAL NETWORK
def regression_model():
    model=Sequential()
    model.add(Dense(1000,activation='relu',input_shape=(n_cols,)))#50 is the number of neurons in eeach layer #relu is the activation function#this code is for first layer because it takes all the data as the neuron from the dataset
    model.add(Dense(1000,activation='relu'))#this is for the hidden layer with 5 neuron each
    model.add(Dense(1))# this is the output layer with one neuron
    #compile model
    model.compile(optimizer='adam',loss='mean_squared_error')
    gc.collect()
    return model
#TRAIN AND TEST THE NETWORK
model=regression_model()
#FIT THE MODEL
model.fit(predictors_norm,target,validation_split=0.3,epochs=100000,verbose=2) #EPOCHS IS SIMPLY THE NUMBER OF TESTS OR THE NUMBER OF TIMES BACKTRACKING IS RUNNED #verbose is the choice that how you want to see the output of your Nural Network while it's training







