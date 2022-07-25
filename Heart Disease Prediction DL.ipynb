#1. Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#2. Data preparation
#load from sklearn
df = pd.read_csv(r"C:\Users\leekt\OneDrive\Documents\GitHub\AI07-Exercises\heart.csv")
df.head()

#Split into x and y
x= df.drop('target',axis=1)
y= df['target']

#%%
SEED = 888
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=SEED)
#%%

standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)
#%%
nClass = len(np.unique(np.array(y_test)))

model = keras.Sequential()

#input layer
model.add(layers.InputLayer(input_shape=(x_train.shape[1],)))
#Nthe hidden layer
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
#output layer ( be careful about the number of nodes and the activation function)
model.add(layers.Dense(nClass, activation='softmax'))

#%%
#Show the structure of the model
model.summary()

#%%
#Compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
#Train model
BATCH_SIZE = 20
EPOCHS = 40
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS)

#%%
#Visualization
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training Accuracy')
plt.plot(epochs_x_axis,val_acc,label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()

#%%
#Evaluate the models accuracy
model_ev = pd.DataFrame({'Model': ['Model 1'], 'Training Accuracy %': [training_acc[-1]*100],'Validation Accuracy %': [val_acc[-1]*100], 'Training Loss %': [training_loss[-1]*100], 'Validation Loss%': [val_loss[-1]*100]})
model_ev