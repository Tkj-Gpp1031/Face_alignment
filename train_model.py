#Import packages
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation,Convolution2D,Flatten,Dense,Dropout,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import cv2
import os
#Check if GPU is consumed
print(tf.__version__)
print(tf.test.is_built_with_cuda)
print(tf.test.is_gpu_available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#Change the train data and test data by changing the following two variables
train_path = 'training_images.npz'
test_path = 'test_images.npz'
# load data
train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)
# Extract the train images
images = train_data['images']
image = images[example_idx]
test_images = test_data['images']
# Extract the data points
train_lm = train_data['points']
#Data process
images = np.array(images)#Ensure data is array
x_train = []#Used to store processed image
x_train_1 = []#Used to store processed data
#Image processing, downscaling and resize ,but in my code i have not use resize
for i in range(images.shape[0]):
    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    img = np.uint8(np.mean(img, axis=-1))
    #img = cv2.resize(img,(96,96))
    x_train.append(img)
x_train = np.array(x_train)
x_train=np.expand_dims(x_train,axis=3)

temp_y = []#Temporary storage of landmark data
y_train = train_lm
y_final = []#new landmark data set
#Iterate through landmark points and turn them into a one-dimensional array
for i in range(y_train.shape[0]):
    temp_y = []
    for j in range(y_train.shape[1]):
        temp_y.append(y_train[i][j][0])
        temp_y.append(y_train[i][j][1])
    y_final.append(temp_y)
y_train = np.array(y_final)#change data type to array
#Building CNN models
#Using the sequence function
model = Sequential()
#Set convolution layer and input the size and number of convolution kernels and determine the shape of the input data
model.add(Convolution2D(32,(3,3),padding= 'same',use_bias= False , input_shape=(244,244,1)))
#Using the activation function LeakyReLU
model.add(LeakyReLU(alpha= 0.1))
#Normative output
model.add(BatchNormalization())

model.add(Convolution2D(32,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(96,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(256,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(512,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512,(3,3),padding= 'same',use_bias= False ))
model.add(LeakyReLU(alpha= 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#one-dimensionalising a multi-dimensional input
model.add(Flatten())
#Define the number of nodes is 512
model.add(Dense(512)) #512
#Enhanced generalization capabilities
model.add(Dropout(0.1))
model.add(Activation('relu'))
#out put 84 point [x1,y1,x2,y1,...,xi,yi]
model.add(Dense(84))
#save model
model.save('final_model/my_model')
#out put on pytharm
model.summary()
#Compilation and print acc
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','acc'])
model.fit(x_train,y_train,batch_size=20,epochs=2000,validation_batch_size=2.0)
#save model
model.save('final_model/my_model')

