from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras



#Load model
new_model = tf.keras.models.load_model('mymodel.h5')
fashion_mnist = keras.datasets.fashion_mnist

# Load Inference data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0

test_images = test_images / 255.0

# Getting model metrics
test_loss, test_acc = new_model.evaluate(test_images,  test_labels, verbose=2)

#Print Test accuracy
print('\nTest accuracy:', test_acc)





