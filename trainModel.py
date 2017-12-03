import gzip
import numpy as np 
import tensorflow as tf

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

train_Labels = labelss("C:\\Users\\Damian Curran\\Desktop\\mnist\\train-labels.gz")

depth = 10

train_Labels_notHot = np.array(train_Labels)
train_Labels_hot = np.eye(depth)[train_Labels_notHot]

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
                row.extend(col)
            images.append(row)     
            
        return images

train_Images = imagess("C:\\Users\\Damian Curran\\Desktop\\mnist\\train-images.gz")

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])

model = (x @ w) + b

y = tf.placeholder(tf.float32, [None, 10])
y_int = tf.placeholder(tf.int64, [None])

y_pred = tf.nn.softmax(model)
y_pred_int = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model,
                                                        labels=y)
cost = tf.reduce_mean(cross_entropy)

correct = tf.equal(y_pred_int, y_int)

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
y_value = y_pred_int

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for i in range(23):
    feed_dict_train = {x: train_Images,
                       y: train_Labels_hot}

    session.run(train, feed_dict=feed_dict_train)

test_Images = imagess("C:\\Users\\Damian Curran\\Desktop\\mnist\\test-images.gz")
test_Labels = labelss("C:\\Users\\Damian Curran\\Desktop\\mnist\\test-labels.gz")

depth = 10

test_Labels_notHot = np.array(test_Labels)
test_Labels_hot = np.eye(depth)[test_Labels_notHot]

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