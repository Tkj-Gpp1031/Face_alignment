# import package
import random
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
def visualise_pts(img, pts):
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r')
  plt.show()
def save_as_csv(points,num,location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==num, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==2*42, 'wrong number of points provided. There should be 42 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
test_path = 'test_images.npz'

#load test data
test_data = np.load(test_path, allow_pickle=True)
# get images data
test_data = test_data['images']
#process data
x_test = []# save processed images
for i in range(test_data.shape[0]):
    img = cv2.cvtColor(test_data[i], cv2.COLOR_BGR2RGB)
    x_test.append(np.uint8(np.mean(img, axis=-1)))
x_test = np.array(x_test)
x_test=np.expand_dims(x_test,axis=3)
print('x_test:',x_test.shape)
# load model that we have train
# new_model = tf.keras.models.load_model('final_model/my_model')
new_model = tf.keras.models.load_model('saved_model/my_model')
#get the predict result
pred = new_model.predict(x_test)
#reshape the result as a Normalising input shapes
pred = pred.reshape(-1,42,2)
# save as csv
save_as_csv(pred,x_test.shape[0])
print('SAVE SUSSEXFULLY！ ')
#read csv
data = pd.read_csv("results.csv",header=None)
data = np.array(data)
#reshape ,enables points to be visualised
data = data.reshape(-1,42,2)
test_data_1 = np.load(test_path, allow_pickle=True)
test_images = test_data_1['images']
for i in range(1):
    # idx = random.randint(0,test_images.shape[0])
    idx = 0
    visualise_pts(test_images[idx, ...], data[idx, ...])
