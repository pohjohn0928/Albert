import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
from numpy import asarray
from PIL import Image
import joblib
from keras.models import load_model
import tensorflow as tf
import joblib




def show_img(arr):
    plt.imshow(arr)
    plt.show()


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print(f'x_train shape : {x_train.shape} {type(x_train)}')
print(f'y_train shape : {y_train.shape} {type(y_train)}')
print(f'x_test shape : {x_test.shape} {type(x_test)}')
print(f'y_test shape : {y_test.shape} {type(y_test)}')

x_train4d = x_train.reshape(x_train.shape[0],28,28,1)
x_test4d = x_test.reshape(x_test.shape[0],28,28,1)

x_train4d_normalize = x_train4d / 255
x_test4d_normalize = x_test4d / 255
print(x_train4d_normalize.shape)

y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

model = Sequential()
model.add(Conv2D(filters=16,
           kernel_size=(5,5),
           padding='same',
           input_shape=(28,28,1),
           activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=26,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x = x_train4d_normalize,
                          y = y_train_one_hot,
                          validation_split=0.2,
                          epochs=1,
                          batch_size=300,
                          verbose=1)
model.save('CNN_model')



# models = load_model('CNN_model')
# image = Image.open('/Users/johnpoh/Desktop/7.png')
# data = np.array([np.asarray(image.resize((28,28)))])
# show_img(data[0])
# data = data.reshape(data.shape[0],28,28,1)
# print(np.argmax(models.predict(data)))


# data = asarray(data)
# data = np.array([data])
# data = data[:,:,:,0]
# data = data[:,:,:,np.newaxis]
# print(models.predict(data))
# data = asarray(data)
# data = data / 255
# show_img(data)
# print(np.argmax(models.predict(x_test4d_normalize), axis=-1))
# print(models.predict(x_test4d_normalize))






