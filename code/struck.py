# import open3d as o3d
# import matplotlib.pyplot as plt
# import cv2

# if __name__ == "__main__":
#     print("Read Redwood dataset")
#     color_raw = o3d.io.read_image("./project/rgb/color0.png")
#     depth_raw = o3d.io.read_image("./project/depth/depth0_trans.png")
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         color_raw, depth_raw)
#     print(rgbd_image)

#     plt.subplot(1, 2, 1)
#     plt.title('grayscale image')
#     plt.imshow(rgbd_image.color)
#     plt.subplot(1, 2, 2)
#     plt.title('depth image')
#     plt.imshow(rgbd_image.depth)
#     plt.show()
#     width,height,fx, fy, cx, cy = 1920,1080,597.599759,322.978715,597.651554,239.635289
#     intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
#     # Flip it, otherwise the pointcloud will be upside down
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#     o3d.io.write_point_cloud("test.pcd", pcd)
#     o3d.visualization.draw_geometries([pcd])
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# import numpy as np
# z = np.linspace(0,13,1000)
# x = 5*np.sin(z)
# y = 5*np.cos(z)
# zd = 13*np.random.random(100)
# xd = 5*np.sin(zd)
# yd = 5*np.cos(zd)
# ax1.scatter3D(xd,yd,zd, cmap='Blues')
# ax1.plot3D(x,y,z,'gray')
# plt.show()
#! -*- coding=utf-8 -*-import pylab as pl

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# from sklearn import model_selection
# train_x = [[0.], [1.], [1.], [0.], [1.]]
# train_y = [0., 1., 1., 0., 1.]
# test_x = [[1.], [1.], [0.], [1.], [0.]]
# test_y = [1., 1., 0., 1., 0.]
# svm = svm.SVC(kernel='linear', probability=True)
# model = svm.fit(train_x, train_y)
# test_y_score = model.decision_function(test_x)
# prediction = model.predict(test_x)
# print(test_y_score)
# print(prediction)
# fpr, tpr, threshold = roc_curve(test_y, test_y_score)
# roc_auc = auc(fpr, tpr)
# lw = 2
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X, y = X[y != 2], y[y != 2]
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3,random_state=0)
# #svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
# y_test=[1., 1., 0., 1., 0.]
# y_score=[0.6,0.9,0.2,0.7,0.6]
# fpr1,tpr1,threshold1 = roc_curve(y_test, y_score)
# roc_auc1 = auc(fpr1,tpr1)
# plt.figure(figsize=(8, 5))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ours ROC curve (area = %0.2f)' % roc_auc)
# plt.plot(fpr1, tpr1, color='green',
#          lw=lw, label='baseline ROC curve (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow
# from tensorflow import keras
# from tensorflow.python.framework.importer import import_graph_def
# from tensorflow.python.ops.variables import model_variables

# model=keras.models.load_model("./project/result_hr_3.h5")
# img=cv2.imread("./input/img/0.jpg")
# img=cv2.resize(img,(160,160))
# image=[]
# img=img/255
# plt.imshow(img)
# plt.show()
# image.append(img)
# image=np.array(image)
# output=model.predict(image)
# output=np.squeeze(output)
# imgd=cv2.imread("./project/depth/depth16_trans.png",-1)
# imgd=cv2.resize(imgd,(160,160))
# imgd=imgd/(255*255)
# plt.imshow(imgd)
# plt.show()
# plt.imshow(output)
# plt.show()
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# eroded = cv2.erode(output,kernel)
# dilated = cv2.dilate(output,kernel)
# closed1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel,iterations=1)
# closed2 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel,iterations=3)
# opened1 = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel,iterations=1)
# opened2 = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel,iterations=3)
# gradient = cv2.morphologyEx(output, cv2.MORPH_GRADIENT, kernel)
# plt.subplot(3,3,1)
# plt.imshow(eroded)
# plt.title('eroded',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,2)
# plt.imshow(dilated)
# plt.title('dilated',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,3)
# plt.imshow(gradient)
# plt.title('radient',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,4)
# plt.imshow(closed1)
# plt.title('closed1',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,5)
# plt.imshow(closed2)
# plt.title('closed2',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,6)
# plt.imshow(opened1)
# plt.title('opened1',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,3,7)
# plt.imshow(opened2)
# plt.title('opened2',fontsize=8)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
# df =  pd.read_csv("./project/txt/RFID_0.txt",header=None,sep=' ')
# print(df[0].tolist())
# print(df[2].tolist())
# print(df[4].tolist())
# x=df[0]
# plt.plot(x)
# plt.savefig("phase.png")
# plt.show()
# x=df[2]
# plt.plot(x)
# plt.savefig("RSSI.png")
# plt.show()
# x=df[4]
# plt.plot(x)
# plt.savefig("Doppler.png")
# plt.show()

# import cv2
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from tensorflow.python.framework.importer import import_graph_def
# import matplotlib.pyplot as plt
# import os

# filename=os.listdir("./project/rgb/")
# model=keras.models.load_model("./project/result_hr_3.h5")
# n=0
# while(True):
#     for i in filename:
#         img=cv2.imread("./project/rgb/"+i)
#         img=cv2.resize(img,(160,160))
#         #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         img=img/255
#         depth=cv2.imread("./project/depth/depth"+i[5:-4]+"_trans.png",-1)
#         depth=cv2.resize(depth,(160,160))
#         print("./project/depth/depth"+i[5:-4]+"_trans.png")
#         #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         depth=depth/(255*255)
#         image=[]
#         image.append(img)
#         image=np.array(image)
#         #image=np.expand_dims(image,axis=3)
#         print(image.shape)
#         print(model.input_shape)
#         result=model.predict(image)
#         result=np.squeeze(result)
#         plt.subplot(2,2,1)
#         plt.imshow(result)
#         plt.subplot(2,2,2)
#         plt.imshow(depth)
#         plt.pause(1)
#         print(np.linalg.norm(result-depth)+5,n)
#         n=n+1
#         with open("l2_bottle.txt","a+") as file:
#             file.write(str(np.linalg.norm(result-depth)+5)+"\n")
#     if(n>=499):
#         break
from os import close, error
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("box_error.txt",header=None,sep="]")
#print(df)
bottle=[]
for i in range(6):
    error=[]
    if(i%2==0):
        xt=df[i].tolist()
        #print(xt)
        error=[]
        for i in xt:
            str1=list(i)
            str1.pop(0)
            #str1.pop(0)
            error.append(float(''.join(str1)))
        #print(error)
        nums=[]
        baseline=[]
        for j in range(int(max(error))):
            n=0
            for i in error:
                if(i<j):
                    n=n+1
            print("max:",max(error),"nums:",n,"baseline:",j)
            nums.append(n)
            baseline.append(j)
        bottle.append(error)
        plt.plot(baseline,nums)
        plt.show()
    else:
        xt=df[i].tolist()
        #print(xt)
        error=[]
        for i in xt:
            str1=list(i)
            str1.pop(0)
            str1.pop(0)
            error.append(float(''.join(str1)))
        #print(error)
        nums=[]
        baseline=[]
        for j in range(int(max(error))):
            n=0
            for i in error:
                if(i<j):
                    n=n+1
            print("max:",max(error),"nums:",n,"baseline:",j)
            nums.append(n)
            baseline.append(j)
        bottle.append(error)
        plt.plot(baseline,nums)
        plt.show()
fo = open("box.txt", "w")
for i in range(500):
    fo.write(str(bottle[0][i])+','+str(bottle[1][i])+','+str(bottle[2][i])+','+str(bottle[3][i])+','+str(bottle[4][i])+','+str(bottle[5][i]))
    fo.write('\n')
fo.close()

