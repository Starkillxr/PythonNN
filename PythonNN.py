import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision, preprocessing, regularizers
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer
from tensorflow import Module
from tensorflow.keras.layers import Layer

# Sets the number of epochs
epoch = 100
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data();
#train_images, test_images = train_images / 255.0, test_images / 255.0

# This is the input images, it pulls from the tensor flow datasets and outputs the fashion mnist dataset
# into the train images, train labels, test images & test labels
# 50,000 train images, 10,000 test images
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# the dimensions of the images
img_rows, img_cols = 28, 28
#These are the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# This is so that the CNN below knows whether its the colour channels first or after the image size and 
# sets the input shape
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape =(1, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Divides the train and test images by 255, I believe this is so that the last channel is either a 1 or a 0 rather than being 234 Red, Green or Blue

train_images = train_images / 255.0
test_images = test_images / 255.0

# Some optimisation code for tensor codes
mixed_precision.set_global_policy('mixed_float16')

# the kernel regularizer variable, there are two different kernel regularizers, l1 and l2. for my NN
# I used the L2 ones, however in future iterations I may use L1 or both of them. Weight regularizers is 
# To help deal with overfit and underfit that the CNN may face. There are two different ways that this helps
# and it depends on the regularizer that is used. for the L2 regularization, it manes that in the weight matrix
# of every layer it will add 0.0001 * weight_coefficient_value **2 https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
kernel_regularizer = regularizers.l2(0.0001)

# This is a custom decay learning rate scheduler that I have coded in order to optimise the CNN and make sure that it
# is continually learning so after epoch 10 the learning rate decays, then 30m then 50, etc.
def decay(epoch):
    if epoch < 11: #original is 11
        return 0.001 #original is 0.001
    elif epoch >= 11 and epoch < 31: #11 to 30
        return 0.0001 #original is 0.0001
    elif epoch>= 31 and epoch < 51: #30 to 64
        return 0.00001 #original is 0.00001
    elif epoch >= 51 and epoch <71: #49 to 68
        return 0.000001
    else:
        return 0.0000001

# This is the data augmentation part of the program, this is to help with training the CNN. The idea behind this is that
# if you add variations in the image such as random rotations, flips and contrast, the CNN will then be able to identify
# the images better when it comes to testing as they will be able to understand the images better instead of just from the
# basic angle that the dataset provides
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  #layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomContrast(0.3)
 
])

# The start of the CNN
model = models.Sequential([
    layers.InputLayer(input_shape = input_shape), # This is the input layer of the CNN
    data_augmentation, # This calls the data augmentation step as seen above
    layers.BatchNormalization(), # This normalises the data before it is put into the CNN
    #model.add(layers.UpSampling2D(size=(3,3), interpolation='nearest'))
    layers.GaussianNoise(stddev=0.02), # This adds noise to the images. This might not have any affect and may make it worse, but it was something I added before I implemented the Data Augmentation section
    layers.Conv2D(280, kernel_size = (7, 7), activation='relu', kernel_regularizer=kernel_regularizer, padding='same'), # This is the first convolution, it has 280 filters of the kernel size 7x7 implementing the activation relu
    # as you can see it also calls the kernel regularizer and the "padding same" variable makes sure that when the convolutions are applied the output image is the same size as the input image.
    #layers.PReLU(),
    layers.MaxPooling2D((2, 2)), #Not entirely sure how this works, I'm assuming the (2,2) part means that it divides each dimension by 2 but I could be wrong, essentiall it pools all the data into a certain area and resizes the image
    layers.BatchNormalization(), # This normalises the data, not sure if it has any effect once it has gone through the first convolution but this is something that I will investigate at a later date
    layers.Dropout(0.25), # This only applies during the training but essentiall what this means is that 25% of the images after this point, this helps prevent overfitting
    layers.Conv2D(560, (7, 7), activation='relu', kernel_regularizer=kernel_regularizer, padding = 'same'), # This has 560 filters of the size 7x7, applies the activation function relu and makes sure the output matrix is the same as the input
    #model.add(layers.PReLU()),
    layers.MaxPooling2D((2,2)), # Pools all the data from the previous convolution together
    layers.BatchNormalization(), # Normalises it, same thing applies not sure if it makes any difference at this stage
    layers.Conv2D(1120, (4, 4), activation='relu', kernel_regularizer=kernel_regularizer), # 1120 filters of the size 4 x 4, I have slowly increase the number of filters per convolution as this is something I have found to be effective
    # when it comes to accuracy
    #model.add(layers.PReLU()),
    layers.MaxPooling2D(2, 2), # Pools all the data together
    layers.BatchNormalization(), # Normalises it
    #model.add(layers.UpSampling2D(size=(3,3), interpolation='nearest')),
    #model.summary(),
    layers.Flatten(), # Flattens the matrix into a matrix with only row
    layers.BatchNormalization(), #Normalises the data
    layers.Dense(512,  activation='tanh', kernel_regularizer=kernel_regularizer), # This dense layer has 512 nodes and implements the activation function tangent
    layers.Dropout(0.5), # 50% of images after the training  portion are dropped out, might find that the overall accuracy increases if removed as this isn't in the part of feature extraction & I have seen no examples in the leaderboards
    # that are on the fashion mnist github that use dropout in the NN part of the CNN.
    layers.BatchNormalization(), # Normalises the data 
    layers.Dense(512,  activation='tanh', kernel_regularizer=kernel_regularizer),# This dense layer has 512 nodes and implements the activation function tangent
    layers.Dropout(0.5), # 50% of the images in the training stage drop out
    layers.BatchNormalization(), # Normalises the data
    layers.Dense(512, activation='tanh', kernel_regularizer=kernel_regularizer),# This dense layer has 512 nodes and implements the activation function tangent
    #layers.Dropout(0.5),
    layers.BatchNormalization(), # Normalises the data
    layers.Dense(512, activation='tanh', kernel_regularizer=kernel_regularizer), # This dense layer has 512 nodes and implements the activation function tangent
    layers.BatchNormalization(), # Normalises the data
    #model.add(layers.Dense(1024, activation='tanh')),
    #model.add(layers.Dropout(0.5)),
    #tf.keras.layers.BatchNormalization(),
    #model.add(layers.PReLU()),
    layers.Dense(1024, activation='swish', kernel_regularizer=kernel_regularizer), # This is a dense layer of 1024 nodes and implements the activation function swish
    #model.add(layers.Dropout(0.5)),
    #model.add(layers.Dropout(0.5)),
    layers.BatchNormalization(), # Normalises the data before output
    #model.add(layers.Dropout(0.3)),
    #model.add(layers.Dense(112, kernel_regularizer='l1_l2', trainable=True, activation='relu')),
    #model.add(layers.Dropout(0.5)),
    #model.add(layers.Dense(32, activation='relu')),
    layers.Dense(10, activation= 'softmax', kernel_regularizer=kernel_regularizer) # this is the output layer, the number of nodes is equal to the number of classes that this dataset has, it implements the softmax activation function
    # as this is the most commonly use activation function on the output layer.
    ])
model.summary() # Summary of the model
# Below compiles the model and uses the optimizer adam within a loss scale optimiser
# The loss is just what was suggested to be used when I first started looking at the tensorflow examples and the metrics part tracks the accuracy of the NN
model.compile(optimizer= tf.keras.mixed_precision.LossScaleOptimizer( tf.keras.optimizers.Adam()),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# This calls the custom learning rate decay that was mentioned earlier
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(decay)]

# This allows for a graph below to be posted, it also mentions how many epochs the program will run, the batch sizes and it also implements the fcustom learning decay rate
history = model.fit(train_images, train_labels, epochs=epoch, batch_size= 200,use_multiprocessing=True, 
                   validation_data=(test_images, test_labels), callbacks=callbacks,  verbose=2)

# Plots a graph of size 10,10 first sub graph is a graph with training and validation accuracy, the second shows the training and validation loss
# You can use this graph to plot many other things, there are a bunch of metrics that tensorflow offers and all of these should be able to be plotted into a graph

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylim([0, 1.0])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()

#This prints the test accuracy after showing & closing/saving the graph
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)