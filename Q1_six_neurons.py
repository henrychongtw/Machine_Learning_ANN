import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()

# a batch of inputs of 8 value each
inputs = tf.placeholder(tf.float32, shape=[None, 8])

# a batch of output of 8 value each
desired_outputs = tf.placeholder(tf.float32, shape=[None, 8])

# [!] define the number of hidden units in the first layer
HIDDEN_UNITS = 6

# connect 8 inputs to 6 hidden units
# [!] Initialize weights with random numbers, to make the network learn
weights_1 = tf.Variable(tf.truncated_normal([8, HIDDEN_UNITS]))

# [!] The biases are single values per hidden unit
biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS]))

# connect 8 inputs to every hidden unit. Add bias
layer_1_outputs = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)

# connect first hidden units to 6 hidden units in the second hidden layer（output layer)
weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, 8]))
# # [!] The same of above
biases_2 = tf.Variable(tf.zeros([8]))

# # connect the hidden units to the second hidden layer(output layer)
logits = tf.nn.sigmoid(
    tf.matmul(layer_1_outputs, weights_2) + biases_2)

# retrieve logits and desired_outputs and see if their argmax matches, return true if yes
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(desired_outputs, 1))
# convert true or false to binary number 0 or 1
acc = tf.cast(correct_prediction, tf.float32)

error_function = 0.5 * tf.reduce_sum(tf.subtract(logits, desired_outputs) * tf.subtract(logits, desired_outputs))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(error_function)

sess.run(tf.initialize_all_variables())

training_inputs = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]

training_outputs = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]

for i in range(500000):
    _, loss = sess.run([train_step, error_function],
                       feed_dict={inputs: np.array(training_inputs),
                                  desired_outputs: np.array(training_outputs)})
    if i % 5000 == 0:
        print('iteration', i)
        print(loss)
        total_acc = 0
        for test in range(len(training_inputs)):
            total_acc += sess.run(acc, feed_dict={inputs: [training_inputs[test]],desired_outputs: [training_inputs[test]]})[0]
        a = (total_acc /len(training_inputs))
        print('accuracy:', a)

print(sess.run(logits, feed_dict={inputs: np.array([[1,0,0,0,0,0,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,1,0,0,0,0,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,1,0,0,0,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,0,1,0,0,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,0,0,1,0,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,0,0,0,1,0,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,0,0,0,0,1,0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[0,0,0,0,0,0,0,1]])}))
