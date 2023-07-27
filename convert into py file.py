#convert to image
import numpy as np
import cv2
import os

y_pred_path = 'stage1_test/images'
x_pred_path = 'stage1_test/masks'
y_pred = np.load('y_predicted.npz')
y_pred = y_pred['arr_0']

x_pred_files = os.listdir(x_pred_path)
x_pred_num = []
for file in x_pred_files:
	x_pred_num.append(int(file.removesuffix(".png")))
x_pred_num = sorted(x_pred_num)

index = 0
for num in x_pred_num:
	y_pred_file = os.path.join(y_pred_path,str(num)+".png")
	y_pred_arr = y_pred[index]
	y_pred_arr = y_pred_arr.astype(np.uint8)
	cv2.imwrite(y_pred_file,y_pred_arr)
	index+=1