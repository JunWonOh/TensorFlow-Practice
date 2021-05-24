# notes taken from tutorial: https://www.youtube.com/watch?v=Zi4i7Q0zrBs

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#import the data set with 6,000 (classified) handwritten digits
mnist = tf.keras.datasets.mnist
#load data from mnist data set. Split it into tuples containing training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#L2-normalizes the given array
#scale it down between 0 and 1 based on RGB percentage (makes training data easier to be computed)
#y values are the labels, (0-9) so it's a bad idea to scale them down
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create a basic neural network
model = tf.keras.models.Sequential()
#flatten creates a 1D layer (turns 28 by 28 image into a size 784 array)
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#units=# neurons wanted in the layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#final output layer
#softmax=gets the probability of correct number
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#loss=the loss function
#metrics=what we're interested in
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#fit the model
#epochs=how many times the model will see the same data over again
model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print('accuracy: ' + str(accuracy))
print('loss: ' + str(loss))

model.save('digits.model')

#interpret hand-written digits
for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()