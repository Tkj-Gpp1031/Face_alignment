#eyes
#0-6 脸部轮廓
#7-9 左meimao
#10-12 右眉毛
#13-17 鼻子
#18-19 左眼
#20-21 右眼
#22-41 嘴巴
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#process data
test_path = 'test_images.npz'#must same as the test_path in file save_visualise.py
test_data_1 = np.load(test_path, allow_pickle=True)
test_images = test_data_1['images']
data = pd.read_csv("results.csv",header=None)
data = np.array(data)
data = data.reshape(-1,42,2)
# The points around the eyes are calculated.
# Make a mask based on the points.
# The mask is then translucent and combined with the image
def get_eyes_mask(idx,side,choose):
    #eyes points distribution(Approximately) :
    '''
                    C
           B                 D
     A              I              E
           H                 F
                    G
    '''
    #Judging the direction of the eye for making masks
    eyes_lst = []
    if side =='left':
        eyes_lst = [18,19]
    else:
        eyes_lst = [20,21]
    eye = data[idx,eyes_lst]

    # left_eye[0] is left eye. left_eye[0][0] is x asix of left eyes
    a_point = eye[0]
    e_point = eye[1]
    a_e_xlen = eye[1][0] - eye[0][0]#Difference of ae horizontal coordinates
    c_g_ylen = a_e_xlen/2 *1.1#Difference of cg horizontal coordinates
    #Calculating the points around the eye and eye length and width
    i_center = [(eye[0][0]+eye[1][0])/2,(eye[0][1]+eye[1][1])/2]
    c_point = [i_center[0] , i_center[1] - c_g_ylen / 2 ]
    g_point = [i_center[0] , i_center[1] + c_g_ylen / 2  ]
    b_point = [a_point[0]+a_e_xlen/4  , a_point[1]- c_g_ylen/4  ]
    d_point = [e_point[0] - a_e_xlen / 4 , e_point[1] - c_g_ylen / 4 ]
    f_point = [e_point[0] - a_e_xlen / 4 , e_point[1] + c_g_ylen / 4 ]
    h_point = [a_point[0] + a_e_xlen / 4 , a_point[1] + c_g_ylen / 4 ]
    eye_length = eye[1][0] - eye[0][0]
    eye_width = (eye_length/2)*1.1
    eye_center = [(eye[0][0]+eye[1][0])/2,(eye[0][1]+eye[1][1])/2]
    #Make a bounding box which can cover the eyes.This step is to be able to capture the eye image
    bounding_box = [[int(eye_center[0] - eye_length / 2),int(eye_center[1] - eye_width / 2)],
                    [int(eye_center[0] + eye_length / 2),int(eye_center[1] - eye_width / 2)],
                    [int(eye_center[0] - eye_length / 2),int(eye_center[1] + eye_width / 2)],
                    [int(eye_center[0] + eye_length / 2),int(eye_center[1] + eye_width / 2)]]
    image = test_images[idx]
    #Intercepted eye image
    img = image[bounding_box[0][1]:bounding_box[2][1], bounding_box[0][0]:bounding_box[1][0]]
    # Change the colour with a transparent mask
    # # build a mask
    zeros = np.zeros(image.shape, dtype=np.uint8)
    #Depending on the input of the function
    #Decide whether to do a colour change of the whole eye or just the corneal part.
    if choose == 'pupil':
        pupil_points = [b_point,d_point,f_point,h_point]
        pupil_points = np.array([pupil_points],dtype=np.int32)
        eye_center = np.array(eye_center,dtype=np.int32)
        eye_mask = cv2.circle(zeros, eye_center, int(a_e_xlen/4), (178,34,34),-1)
    else:
        #Coordinates of points on the outside of the mask
        eye_points = [a_point,b_point,c_point,d_point,e_point,f_point,g_point,h_point]
        eye_points = np.array([eye_points], dtype=np.int32)
        eye_mask = cv2.fillPoly(zeros, pts=eye_points, color=(0, 165, 255))

    return eye_mask
def get_lips_mask(idx,side):
    lips = []
    #Find the coordinates of the edge of the mouth to facilitate the creation of the mask

    if side =='up':
        lips_up_lst = [22,23,24,25,26,27,28]#上嘴唇的上半边
        lips_down_lst=[35,36,37,38]#上嘴唇的下半边
    else:
        lips_down_lst = [28,29,30,31,32,33,34]#下嘴唇的下半边
        lips_up_lst = [38,39,40,41]#下嘴唇的上半边
    image = test_images[idx]
    #because the lips landmark are not continuous. So it is divided into two lists
    lips_up = data[idx, lips_up_lst]
    lips_down = data[idx, lips_down_lst]
    points = []
    for i in lips_up:
        points.append(i)
    for j in lips_down:
        points.append(j)
    points = np.array([points], dtype=np.int32)
    zeros = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.fillPoly(zeros, pts=points, color=(250,128,114))
    return mask


for i in range(6):
    left_mask = get_eyes_mask(i,'left','pupil')
    right_mask = get_eyes_mask(i,'right','all')
    lips_up_mask = get_lips_mask(i,'up')
    lips_down_mask = get_lips_mask(i,'down')
    #Add the all masks to get all the masks to be discoloured
    mask = right_mask + left_mask + lips_up_mask + lips_down_mask
    image = test_images[i]
    alpha = 0.8
    beta = 1 - alpha# mask Transparency
    gamma = 0
    img_add = cv2.addWeighted(image, alpha, mask, beta, gamma)

    plt.imshow(mask)
    plt.show()
    plt.imshow(img_add)
    plt.show()



'''
#Edge detection using the canny operator, 
#followed by processing using morphology to obtain mask. Perform eye discolouration
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

# cv2.imshow("img", img)
plt.imshow(img)
plt.show()


eye = data[0,[18,19]]
# left_eye[0] 是左眼 left_eye[0][0] 是左眼横坐标
eye_length = (eye[1][0] - eye[0][0])*1.4
eye_width = (eye_length/2)*1.1
eye_center = [(eye[0][0]+eye[1][0])/2,(eye[0][1]+eye[1][1])/2]
print('l',eye[0],'r',eye[1],'c',eye_center)
bounding_box = [[int(eye_center[0] - eye_length / 2),int(eye_center[1] - eye_width / 2)],
                [int(eye_center[0] + eye_length / 2),int(eye_center[1] - eye_width / 2)],
                [int(eye_center[0] - eye_length / 2),int(eye_center[1] + eye_width / 2)],
                [int(eye_center[0] + eye_length / 2),int(eye_center[1] + eye_width / 2)]]
print('1',bounding_box)
image = test_images[0]
img = image[bounding_box[0][1]:bounding_box[2][1], bounding_box[0][0]:bounding_box[1][0]]

print(img.shape)
#canny
img1 = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img1, 50, 150)



k = np.ones((5, 5), np.uint8)
close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k)
print(close.shape[0])
cv2.imshow("original_img", close)

cv2.imshow('Canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.imshow(binary)
# plt.show()
change_color = []
for i in range(close.shape[0]):
    for j in range(close.shape[1]):
        if close[i][j] == 255:
            change_color.append([bounding_box[0][1]+i,bounding_box[0][0]+j])
change_color = np.array(change_color)
print("需改变颜色的坐标点：",change_color[17])

for i in range(len(change_color)):
    image[change_color[i][0],change_color[i][1]] = [255,218,185]


# image[ bounding_box[0][1]:bounding_box[2][1],bounding_box[0][0]:bounding_box[1][0]] = close
#
#
plt.imshow(image)
plt.show()
'''
