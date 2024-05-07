import numpy as np
import random
import tensorflow as tf

LEARNING_RATE = 0.001
# Equation as : Y = 2X

# The training data
X = np.array([2.0,4.0,5.0,1.0,3.0,2.0,3.0,4.0])
y = X*2

# The weights and biases

w = tf.Variable(random.random(),trainable= True)
b = tf.Variable(random.random(),trainable= True)

def simple_loss(y_true,y_pred):
    return tf.abs(y_true-y_pred)


def fit_train(X,y,epoch):
    with tf.GradientTape(persistent = True) as tape:

        y_pred = w * X + b

        loss = simple_loss(y,y_pred)

    w_grad = tape.gradient(loss,w)
    b_grad = tape.gradient(loss,b)

    w.assign_sub(w_grad*LEARNING_RATE)
    b.assign_sub(b_grad*LEARNING_RATE)
    
    if epoch % 10 == 0 :
        print(f"The current_weight is : {w.numpy()} and bias is : {b.numpy()}")

for i in range(1000):
    fit_train(X,y,i)

