# improves accuracy of barebones.py
# documentation found here: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow import reshape
from tensorflow.keras.layers import Dense, Input

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = reshape(x_train, shape=[-1, 784]), reshape(x_test, shape=[-1, 784])
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

batch_size = 20
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

model = keras.Sequential([
    Input(shape=784),
    Dense(30, activation="relu"),
    Dense(10, activation="relu"),
])

model.compile(metrics=["accuracy"])

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.MSE

epochs = 5
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for x_batch_train, y_batch_train in train_dataset:
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        # Basically updates weights and biases
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

loss, accuracy = model.evaluate(x_test, y_test)
print("Testing Accuracy: %" + str(100*accuracy))