import gzip
import numpy as np 
import tensorflow as tf

# Adapted from: https://github.com/emerging-technologies/keras-iris and https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#5

#function which reads labels when fed file path and returns 1d array
def labelss(filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = []
            
            #Saves variables that arent bytes for labels
            magicNum = (int.from_bytes(f.read(4), "big"))
            noLabels = (int.from_bytes(f.read(4), "big"))
            
            #for loop for range, 60,000/10,000
            for i in range(noLabels):
                #Reads byte at a time and appends to list
                labels.append((int.from_bytes(f.read(1), "big")))
        
        return labels


#append vs extend:  https://stackoverflow.com/questions/252703/difference-between-append-vs-extend-list-methods-in-python
#function which reads images when fed file path and returns 2d array
def imagess(filepath):
    with gzip.open(filepath, 'rb') as f:     
        
        #Save variables that are not image bytes
        magicNum = (int.from_bytes(f.read(4), "big"))
        noImages = (int.from_bytes(f.read(4), "big"))
        noRow = (int.from_bytes(f.read(4), "big"))
        noCol = (int.from_bytes(f.read(4), "big"))
                
        images = []
        
        for i in range(noImages):
            row = []
            for k in range(noRow):
                col = []
                for j in range(noCol):
                    col.append(int.from_bytes(f.read(1), "big"))
                #extend used here to add each 28 bytes to 1d array of length [1,2,3,4,5,6]
                #vs append which would create a 2d array [[1,2,3],[4,5,6]]
                row.extend(col)
            images.append(row)     
            
        return images

#calls function and feeds it a directory to read file from
train_Labels = labelss("C:\\Users\\Damian Curran\\Desktop\\mnist\\train-labels.gz")

#calls function and feeds it a directory to read file from
train_Images = imagess("C:\\Users\\Damian Curran\\Desktop\\mnist\\train-images.gz")

#np.eye: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.eye.html
#one hot encoding explanation: https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

#setting depth of one_hot, we have 10 labels
depth = 10

#converts to one_hot vector using numpy
train_Labels_hot = np.eye(depth)[train_Labels_notHot]


#These are changed to pythons liking to more accurately adjust towards the correct output
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#this takes as many values of size 784, it is size 784 because we feed it an image array which is length 784
x = tf.placeholder(tf.float32, [None, 784])

#stores the unscaled matrix multiplication into model
model = (x @ w) + b

#this will hold the ouputs, the 10 different label types [0,1,2,3,4,5,6,7,8,9]
y = tf.placeholder(tf.float32, [None, 10])

#this represents the index of label types
y_int = tf.placeholder(tf.int64, [None])

#saves normalized data into y_pred
y_pred = tf.nn.softmax(model)

#gets largest index e.g [1,3,2] largest index = 1
#index representation   [0,1,2]
y_pred_int = tf.argmax(y_pred, axis=1)

#can seperate out softmax and cross_entropy
#using this function is more accurate, and less lines of code
#tf.nn.softmax VS tf.nn.softmax_cross_entropy_with_logits https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model,
                                                        labels=y)
#computes the mean of elements across a tensor
cost = tf.reduce_mean(cross_entropy)

#tf.equal returns bools(true, false) and stores in correct_prediction
correct = tf.equal(y_pred_int, y_int)

#we then cast correct_prediction to a float which returns 0 if false and 1 if true 
#we then use reduce_mean function to find avrage
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#this will be used later to get the predicted int value of our labels
y_value = y_pred_int

#every tensorflow program uses an optimizer, the most used one is GradientDescent
#https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
#searching for best learning rate: https://stackoverflow.com/questions/43851215/optimise-tensorflow-learning-rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

session = tf.Session()
#Initializes all previous mentioned variables, tensors
session.run(tf.global_variables_initializer())

#loop for training the model
for i in range(23):
    #data is fed into model using this construct
    feed_dict_train = {x: train_Images,
                       y: train_Labels_hot}
    
    #calls session.run and runs model method to begin training
    session.run(train, feed_dict=feed_dict_train)

#calls functions to acquire test labels and images
test_Images = imagess("C:\\Users\\Damian Curran\\Desktop\\mnist\\test-images.gz")
test_Labels = labelss("C:\\Users\\Damian Curran\\Desktop\\mnist\\test-labels.gz")

#converts test labels to one_hot
test_Labels_notHot = np.array(test_Labels)
test_Labels_hot = np.eye(depth)[test_Labels_notHot]

#create construct to feed test data
feed_dict_test = {x: test_Images,
                  y: test_Labels_hot,
                  y_int: test_Labels}

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def gotValue(feed_dict_test1):
    value = session.run(y_value, feed_dict=feed_dict_test1)
    return value

def setImage(images):
    feed_dict_test1 = {x: images[0:1]}
    pred = gotValue(feed_dict_test1)
    return pred