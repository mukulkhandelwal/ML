import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
input > weight > hidden layer1 (activation function) > weights ? hidden l 2
(activation function) >weights > output layer


compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimizer cost (AdamOptimizer....SGD, AdaGrad)

backpropagation


feed forward + backprop = epoch

'''


mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]

'''
# 3 hidden layer
#no compulsory to be same

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100  #100 features and manipulate the weight

#placeholder
#matrix = height * width
#28*28 = 784 pixel
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')


def neural_network_model(data):

    #input_data * weights + biases

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal(n_nodes_hl2))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal(n_nodes_hl3))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' : tf.Variable(tf.random_normal(n_classes))}


    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) + hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)  #rectified linear Activation Function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output













