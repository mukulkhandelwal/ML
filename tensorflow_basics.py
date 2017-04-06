import tensorflow as tf


x1 = tf.constant(5)
x2 = tf.constant(6)

#result = x1 * x2 #not a good way
result = tf.multiply(x1,x2)  #tf.matmul  if x1 = tf.constant([5])

print(result)

#
# sess = tf.Session() #begins session
# print(sess.run(result))
# sess.close()

# good way to do not have to close session
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
    print(sess.run(result))


#print(output) #not printed


