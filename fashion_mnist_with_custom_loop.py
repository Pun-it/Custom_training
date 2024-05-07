import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras import Model
import tensorflow_datasets as tfids
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

EPOCHS = 2

# The Model

def base_model():
    inputs = Input(shape = (784,))
    dense1 = Dense(64,activation = "relu")(inputs)
    out = Dense(10,activation = 'softmax')(dense1)

    model = Model(inputs,out)

    return model

model = base_model()


# Data prep

train_data = tfids.load('fashion_mnist',split = 'train')
test_data = tfids.load('fashion_mnist',split = 'test')

def format_image(data):
    image = data['image']
    image = tf.reshape(image,[-1])
    image = tf.cast(image,'float32')
    image = image/255.0

    return image,data['label']

train_data = train_data.map(format_image)
test_data = test_data.map(format_image)

batch_size = 64

train = train_data.shuffle(buffer_size = 1024).batch(batch_size)
test = test_data.batch(batch_size)

# Defining the loss and the optimizer

loss_object = SparseCategoricalCrossentropy()
optimizer_object = Adam()

train_metric = SparseCategoricalAccuracy()
val_metric = SparseCategoricalAccuracy()

# One that does all the grad taping and stuff

def apply_gradients(model,optimizer_object,x,y):
    with tf.GradientTape(persistent= True) as tape:
        logits = model(x)
        loss_value = loss_object(y,logits)
    gradients = tape.gradient(loss_value,model.trainable_variables)
    optimizer_object.apply_gradients(zip(gradients,model.trainable_variables)) 

    return logits,loss_value

# Calculates the losses
def loss_per_epoch():
    losses = []

    for step, (x_batch_train,y_batch_train) in tqdm.tqdm(enumerate(train)):
        
        logits, loss_value = apply_gradients(model,optimizer_object,x_batch_train,y_batch_train)

        train_metric.update_state(y_batch_train,logits)

        losses.append(loss_value)

        # print(f"Number_of_steps : {step},  batch_loss : {loss_value},  batch_accuracy : {train_metric.result()}")
    
    return losses

# losses but for validation
def val_loss_per_epoch():
    losses = []
    for x_val,y_val in test:
        
        logits = model(x_val)

        loss_value = loss_object(y_val,logits)

        val_metric.update_state(y_val,logits)

        losses.append(loss_value)

    return losses


def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
  plt.title(title)
  plt.ylim(0,ylim)
  plt.plot(train_metric,color='blue',label=metric_name)
  plt.plot(val_metric,color='green',label='val_' + metric_name)
  plt.show()

def save_metrics(train_metric, val_metric, metric_name, title, ylim=5):
  plt.title(title)
  plt.ylim(0,ylim)
  plt.plot(train_metric,color='blue',label=metric_name)
  plt.plot(val_metric,color='green',label='val_' + metric_name)
  plt.legend(loc="upper left")
  plt.savefig(f"{title}.png")


# Training : )

def training():
    all_train_losses,all_val_losses = [],[]
    all_train_acc,all_val_acc = [],[]
    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_losses = loss_per_epoch()
        val_losses = val_loss_per_epoch()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)

        train_acc = train_metric.result()
        train_metric.reset_state()

        val_acc = val_metric.result()
        val_metric.reset_state()

        print(f""" EPOCH : {epoch}   TRAINING LOSS : {train_loss}   TRAINING ACCURACY : {train_acc}  VALIDATION LOSS : {val_loss}   VALIDATION_ACCURACY : {val_acc}""")
    
    # save_metrics(all_train_losses, all_val_losses, "Loss", "Loss", ylim=1.0)
    # save_metrics(all_train_acc,all_val_acc,"Accuracy","Accuracy",ylim=1.0)
    # USE in jupyter notebook 
    # plot_metrics(all_train_losses, all_val_losses, "Loss", "Loss", ylim=1.0)
    # plot_metrics(all_train_acc,all_val_acc,"Accuracy","Accuracy",ylim=1.0)


training() # YAY!




