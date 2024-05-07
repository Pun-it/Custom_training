from typing import Any
import tensorflow as tf

# Made by following the Course 2 in the Advanced Tensorflow Specialization by Deeplearning.ai

# Define the Model

class Model():
    def __init__(self):
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    def __call__(self,x):
        return tf.add(tf.multiply(self.w,x),self.b)
    
# Get the Training Data

TRUE_W = 9.0
TRUE_B = 4.0
NUM_DATAPOINTS = 2000

X_train = tf.random.normal(shape=[NUM_DATAPOINTS])

y_train = TRUE_W*X_train+TRUE_B

# Define the loss function
# MSE

def loss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

# Defining the training loop : ) Yaay !

def train(model,inputs,outputs,learning_rate):

    total_loss = [] # Stroing the total loss
    total_params = [[],[]] # Stroing the params
    
    with tf.GradientTape(persistent=True) as tape:

        current_loss = loss(outputs,model(inputs))

    total_loss.append(current_loss)

    total_params[0].append(model.w)
    total_params[1].append(model.b)

    dw,db = tape.gradient(current_loss,[model.w,model.b]) # The classic grad calculation

    # The params getting updated  : )
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)

     # Print the loss
    print(f"current_loss : {current_loss}")

# Training Time

model = Model()
for i in range(20):
    train(model,X_train,y_train,0.1)
