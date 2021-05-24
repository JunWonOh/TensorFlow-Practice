import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

mnist = tf.keras.datasets.mnist
# load the data set
# mnist data set is a tuple of 2 tuples, each with their own 2 entries
# x_train and x_test are 28x28 images representing numbers
# y_train and y_test are the expected numbers (a.k.a, the labels (from 0-9))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# RGB value is between 0 and 255
# by dividing by 255, it converts RGB value to a ratio between 0 and 1
# another way to do it is normalizing the values:
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train, x_test = x_train/255.0, x_test/255.0

# (as opposed to Functional()) the model is just a stack of layers
# could build model inside Sequential function (similar to compile), but chose to use
# model.add()
model = tf.keras.Sequential()
# x_train's shape is (60,000, 28, 28), and x_test is (10,000, 28, 28)
# reshape function: -1 gets for x_train (60,000), x_test (10,000)
# 784 is just 28x28. The desired/updated shape is 60,000x784 for x_train. 10,000x784 for x_test
# alternative line:
# x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dim(x_test, axis=-1)
# the difference being, rather than using a 784 length array for each image, the one above is 28x28.
# Flatten function will be required for above (model.add(tf.keras.layers.Flatten(input_shape=(28,28))))
x_train, x_test = tf.reshape(x_train, shape=[-1, 784]), tf.reshape(x_test, shape=[-1, 784])
# creates a "0" array of size 10, where the 1 takes the index of the desired output (0-9)
# length 60,000 for y_train, 10,000 for y_test
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
# the first layer has 784 nodes
model.add(Input(shape=784))
# two hidden layers. Uses relu to compress values between 0 and 1 rather than sigmoid
# first arg is the no. of perceptrons - the size of layer_n-1 be >= layer_n.
model.add(Dense(30, activation="relu"))
model.add(Dense(20, activation="relu"))
# the output layer (size = 10 because possible output is 0-9)
model.add(Dense(10, activation="relu"))

model.compile(optimizer='adam',                         # could also use SGD (stochastic gradient descent)
              # loss function determines how poorly the model performed
              loss='MSE',                               # could also use sparse_categorical_crossentropy
              metrics=['accuracy'])                     # interested in how accurately numbers are interpreted
# train the model. Pass in the newly formatted images (lines 14 and 24).
# we use a batch size of 10; meaning no. of data in a training set evaluated before updating hyper parameters
# epochs = the number of times the program goes through the training set
model.fit(x_train, y_train, batch_size=10, epochs=10)

# after model has been trained, perform the actual test
# evaluate returns a size 2 array, 1st element being loss, 2nd being accuracy
loss, accuracy = model.evaluate(x_test, y_test)

print("Testing Accuracy: %" + str(100*accuracy))