# Jaromir Panas 10.03.2020
# Code based on the course:
# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
# coursera.org
# Training neural network on the dataset of hand-written digits -- MNIST.
# No regularization and crossvalidation only on test set (no dev set).
# Accuracy reached for the training set approx. 99.5%
# Accuracy reached for the test set approx. 98.9%

import tensorflow as tf
import numpy as np
from mnist import MNIST

#MyCallback class. Inherits from Callback class of tf.keras.callbacks
#Redefines the function on_epoch_end to stop when required accuracy reached
class MyCallback(tf.keras.callbacks.Callback):	
	def on_epoch_end(self, epoch, logs={}):
		#Also check the performance on the test set, to get the feeling of variance/bias ratio
		print("\nChecking performance on the test set.")
		self.model.evaluate(test_img,test_lbl,verbose=2)
		#Check for the accuracy to be above limit of 99.5%.
		if(logs.get('sparse_categorical_accuracy')>0.995):
			print("\nAccuracy reached >99.5%.\n")
			self.model.stop_training=True

#Declare a callback
callback=MyCallback()

# Specify the dataset form tensorflow (MNIST -- hand written digits
mndata=tf.keras.datasets.mnist

# Load the training and test data from the dataset
(train_img,train_lbl),(test_img,test_lbl)=mndata.load_data()

# Normalize the input values to interval [0.,1.]
train_img = train_img/255.
test_img = test_img/255.

# Convolutional layer requires data in format of a 4-dimensional tensor.
# Reshape the 3-dim data to fit the required format using the reshape from numpy.
train_shape=train_img.shape
train_img=train_img.reshape(train_shape[0],train_shape[1],train_shape[2],1)
test_shape=test_img.shape
test_img=test_img.reshape(test_shape[0],test_shape[1],test_shape[2],1)

# Define the model on a machine learning algorithm.
# Here we use sequential neural network.
# To obtain better results we first pass the data through convolutional and max-pooling layers.
# This highlights important features and reduces imput to dense layers.
# Layer flatten ensures compatibility of convlitutional and dense layers.
# Last layer is responsible for softmax classification.
model = tf.keras.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
							tf.keras.layers.MaxPooling2D(2,2),
							tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
							tf.keras.layers.MaxPooling2D(2,2),
							tf.keras.layers.Flatten(),
							tf.keras.layers.Dense(64,activation=tf.nn.relu),
							tf.keras.layers.Dense(10, activation=tf.nn.softmax)
							])
		
# Compile the model of a NN. Use Adam optimization, categorical crossenropy.
# Metrics allows us to acces accuracy in callbacks.
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Print the summary of model: number of layers, types of layers, number of fitting parameters, ...
model.summary()

# Train the NN.
model.fit(train_img,train_lbl,epochs=20,callbacks=[callback])

# Test the model on a test set to check for variance.
print("Test loss and accuracy\n")
test_loss=model.evaluate(test_img,test_lbl,verbose=2)
#print("Test loss and accuracy\n")


input("""Press Enter to continue...""")