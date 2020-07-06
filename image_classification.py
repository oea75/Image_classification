# load libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# load the data
from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# look at the data types of variables
print('Datatypes: ')
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# get the shape of the arrays
print('Shape of the array: ')
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

# take a look at the first image as an array
print('First image as an array: ')
index = 10
x_train[index]

# show image as a picture
print('Image: ')
img = plt.imshow(x_train[index])

# get image label
print('The image label is: ', y_train[index])

# get image classification
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# print image class
print('The image class is: ',classification[y_train[index][0]])

# convert the labels into a set of 10 numbers to input int the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
 
# print the new label of the image above
print('The one hot label is', y_train_one_hot[index])

# normalize the pxels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255

# create the models architecture 
model = Sequential()

# add the first layer
#convolution layer : extract features from input image
model.add(Conv2D(32, (5,5), activation ='relu', input_shape = (32,32,3)) )

# add pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# add a second convolution layer
model.add(Conv2D(32, (5,5), activation ='relu') )

# add another pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

# add a flattening layer to reduce the dimensionality to a linear array
model.add(Flatten())

# add a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

# add dropout layer
model.add(Dropout(0.5))

# add a layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

# add dropout layer
model.add(Dropout(0.5))

# add a layer with 250 neurons
model.add(Dense(  250, activation = 'relu'))

# add a layer with 10 neurons
model.add(Dense(  10, activation = 'softmax'))

# compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# train the model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)

# evaluate the model using the test dataset
model.evaluate(x_test, y_test_one_hot)[1]

# vizualize the model accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','val'], loc = 'upper left')
plt.show()

# vizualize the model loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','val'], loc = 'upper right')
plt.show()

# test model with an example
new_image = plt.imread('cat.jpg')
img = plt.imshow(new_image)

# resize the image 
from skimage.transform import resize
resized_img = resize(new_image,(32,32,3))
img = plt.imshow(resized_img)

# get the models predictions
predictions = model.predict(np.array([resized_img]))

#show the predictions
print('Predictions: ')
predictions

# sort the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

# show the sorted labels in order
print('Prediction from least to greatest: ',list_index)

# print the 5 first predictions 
for i in range(5):
  print('Classification per percentage\n',classification[list_index[i]],':', round(predictions[0][list_index[i]]* 100,2), '%')
  
