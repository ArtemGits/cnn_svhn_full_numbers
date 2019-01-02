import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image
import numpy as np
import time
import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Lambda,Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard


def subtract_mean(a):
    """ Helper function for subtracting the mean of every image
    """
    for i in range(a.shape[0]):
        a[i] -= a[i].mean()
    return a


#preparing the y data
def y_data_transform(y):
    y_new=np.zeros((y.shape[0],y.shape[1]*11),dtype="int")
    for (i,j),l in np.ndenumerate(y):
        y_new[i,j*11+l]=1
    return y_new


_EPSILON=1e-7
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)
def loss_func(y):
    y_pred,y_true=y
    loss=_loss_tensor(y_true,y_pred)
    return loss

def convert_to_num(x):
    num=""
    if len(x)==55:
        for i in range(5):
            c=np.argmax(x[i*11:(i+1)*11])
            if c!=10:
                num+=str(c)
        return num
    else:
        print("This function might not be used that way")



K.clear_session()

h5f = h5py.File('data/SVHN_multi_grey.h5','r')

# Extract the datasets
x_train = h5f['train_dataset'][:]
y_train = h5f['train_labels'][:]
x_val = h5f['valid_dataset'][:]
y_val = h5f['valid_labels'][:]
x_test = h5f['test_dataset'][:]
y_test = h5f['test_labels'][:]

# Close the file
h5f.close()

print('Training set', x_train.shape, y_train.shape)
print('Validation set', x_val.shape, y_val.shape)
print('Test set     ', x_test.shape, y_test.shape)

X_train = np.concatenate([x_train, x_val])
Y_train = np.concatenate([y_train, y_val])

# Randomly shuffle the training data

X_train, Y_train = shuffle(X_train, Y_train)

# Subtract the mean from every image
X_train = subtract_mean(X_train)
X_test = subtract_mean(x_test)

Y_Train=y_data_transform(Y_train)
Y_test=y_data_transform(y_test)


input_data=Input(name="input",shape=(32,32,1),dtype='float32')
conv1=Conv2D(32,5,padding="same",activation="relu")(input_data)
conv2=Conv2D(64,5,padding="same",activation="relu")(conv1)
max1=MaxPooling2D(pool_size=(2, 2),padding="same")(conv2)
drop1=Dropout(0.15)(max1)

#conv3=Conv2D(64,5,padding="same",activation="relu")(drop1)
#conv4=Conv2D(64,5,padding="same",activation="relu")(conv3)
#max2=MaxPooling2D(pool_size=(2, 2),padding="same")(conv4)
#drop2=Dropout(0.15)(max2)

#conv5=Conv2D(128,5,padding="same",activation="relu")(drop1)
#conv6=Conv2D(128,5,padding="same",activation="relu")(conv5)
#conv7=Conv2D(128,5,padding="same",activation="relu")(conv6)
flat=Flatten()(drop1)

#fc1=Dense(64,activation="relu")(flat)
#drop3=Dropout(0.5)(fc1)
#fc2=Dense(253,activation="relu")(drop3)
output=Dense(55,activation="sigmoid")(flat)

#model=Model(inputs=input_data, outputs=output)

y_true = Input(name='y_true', shape=[55], dtype='float32')
loss_out = Lambda(loss_func, output_shape=(1,), name='loss')([output, y_true])
model = Model(inputs=[input_data,y_true], outputs=output)
model.summary()
model.add_loss(K.sum(loss_out,axis=None))

#tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss=None, optimizer="RMSprop", loss_weights=None, metrics=['accuracy'])
model.fit(x=[X_train,Y_Train],y=None, batch_size=1000, epochs=15)

Accuracy=(1-np.mean(model.predict([X_test[:],Y_test[:]])))*100
print(Accuracy)


model.save("MDR_model.h5")
model.save_weights("MDR_model_weights.h5")


X1=model.predict([X_test, Y_test])
#print(X1)
Y1=Y_test
j=0
for i in range(len(X_test)):
    try:
        
        if eval(convert_to_num(X1[i]))!=eval(convert_to_num(Y1[i])):
            j+=1
            #print(i,[convert_to_num(X1[i]),convert_to_num(Y1[i])])
    except:
        j+=1
print("total error",j," out of ",len(X1),"and total accuracy",(1-(j/len(X1)))*100)
