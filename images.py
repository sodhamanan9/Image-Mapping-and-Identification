import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import copy

X = np.linspace(-1, 1, 256)
Y = np.linspace(-1, 1, 256)
x, y = np.meshgrid(X, Y)
# x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)

def gauss(mu, sigma, c):
  return c*(np.exp(-((d-mu)**2 / (2.0 * sigma**2))))
mat1=gauss(1.0, 1.0, 3)
print(mat1)

def mask_gauss(mat):
  mask = np.zeros(mat.shape[:2], dtype="uint8")
  tmp_mat=copy.deepcopy(mat)
  tmp=int(int(len(mat[0]))/int(2))
  for i in range(len(mat)):
    for j in range(tmp):
      mask[i][j]=1
  for i in range(len(mat)):
    for j in range(len(mat[0])):
      if(mask[i][j]!=1):
        tmp_mat[i][j]=0
  return tmp_mat

def gen_c_val(n):
  c_val=[]
  for i in range(n):
    cv=np.random.randint(1, 500)
    c_val.append(cv)
    print(c_val)
  return c_val


X_path = r'C:\All Data\Studies\BTP\X'
Y_path = r'C:\All Data\Studies\BTP\Y'
X_test_path = r'C:\All Data\Studies\BTP\X_test'
X_train_path = r'C:\All Data\Studies\BTP\X_train'
Y_test_path = r'C:\All Data\Studies\BTP\Y_test'
Y_train_path = r'C:\All Data\Studies\BTP\Y_train'

mat2=mask_gauss(mat1)
# print(mat2)

# plt.imshow(mat2)
plt.gray()
plt.imshow(mat2)
plt.show()

c_vals=gen_c_val(50)
print(len(c_vals))
matrices=[]
count = 0
for i in c_vals:
  for mu in range(1,6):
    for sigma in range(1,6):
      mat=gauss(mu, sigma, i)
      temp_mat=mat
      max=np.amax(temp_mat)
      temp_mat=(temp_mat/max)
      temp_mat=temp_mat*255
      img_path=os.path.join(Y_path,str(count)+'.png')
      cv2.imwrite(img_path,temp_mat)
      matrices.append(mat)
      count+=1
print(count)

count=0
masked_mat_gray=[]
for i in matrices:
  masked=mask_gauss(i)
  masked_mat_gray.append(masked)
  temp_mask=masked
  max=np.amax(temp_mask)
  temp_mask=(temp_mask/max)
  temp_mask=temp_mask*255
  img_path=os.path.join(X_path,str(count)+'.png')
  cv2.imwrite(img_path,temp_mask)
  count+=1




