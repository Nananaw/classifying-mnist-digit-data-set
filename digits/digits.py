import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

#changes a digit into a one_hot vector
def toOneHot(number):
    vec = np.zeros(10);
    vec[number] = 1
    return vec

#changes a probability vector (or one_hot) into the digit it represents
def toDigit(vector):
    return np.argmax(vector)    

#load the data set
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#preprocessing
#changes the digits(labels) into their coressponding one_hot vectors 
train_y = np.array([toOneHot(num) for num in train_y])
test_y = np.array([toOneHot(num) for num in test_y])

#creating neural network model
model = tf.keras.Sequential([
    
    #first covolution and pooling layer
    tf.keras.layers.Conv2D(20, (4,4) , activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    
    #second convolution and pooling layer
    tf.keras.layers.Conv2D(40, (4,4) , activation = 'relu'),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    
    #classifying layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

#loss function and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
op = tf.keras.optimizers.Adam()

#compile our model
model.compile(loss = loss_fn, optimizer = op, metrics=['accuracy'])

#fit model to our data
model.fit(train_X, train_y, epochs = 5)

#find accuracy of our model using test data
y_hat = list(map(toDigit, model.predict(test_X)))
y = list(map(toDigit, test_y))
acc = accuracy_score(y, y_hat)

#print accuracy of our model
print('Accuracy on the test set was : ')
print(acc)


