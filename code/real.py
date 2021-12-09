from ctypes import pointer
import sys
import os
import cv2 
import numpy as np
from numpy.lib.type_check import *
import pandas
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import optimizers
import keras
import pandas as pd
from keras import backend as K
from numba import cuda
from multiprocessing import Process,Queue
from tensorflow.python.ops.array_ops import rank_internal
from tensorflow.python.ops.gen_array_ops import depth_to_space
import math

def predict_point(test,q):
    model=load_model("./project/result_u.h5")
    test=np.array(test)
    print(test.shape)
    img1=model.predict(test)
    img1=np.squeeze(img1)
    img1=img1.reshape(4,160,160)
    q.put([img1])

def p2p3(p2,p3,kp,dp):
     index = np.unravel_index(kp.argmax(), kp.shape)
     p2.append(index[0])
     p2.append(index[1])
     p3.append(index[0])
     p3.append(index[1])
     p3.append(dp[index[0]][index[1]]*255*255)


def predict_depth(txts,inputs,q):
    model=load_model("./project/result.h5")
    image=np.array(inputs)
    txts=np.array(txts)
    txts=np.expand_dims(txts,axis=2)
    txts=txts.reshape(1,3,132)
    result=model.predict([image,txts])
    result=np.squeeze(result)
    q.put([result])


if __name__=='__main__':
    depth=cv2.imread("./project/depth/depth100_trans.png",-1)
    depth=cv2.resize(depth,(160,160))
    plt.imshow(depth)
    plt.show()
    s = cv2.FileStorage()
    s.open('./opencv_temp.json', cv2.FileStorage_READ)
    point3d = s.getNode('vector p3').mat()
    #point3d=point3d.reshape(3,4)
    print(point3d,point3d.shape)
    s = cv2.FileStorage()
    s.open('./mtx.json', cv2.FileStorage_READ)
    mtx = s.getNode('mtx').mat()
    print(mtx,mtx.shape)
    img=cv2.imread("./project/rgb/color100.png")
    image=[]
    img=cv2.resize(img,(160,160))
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img/255
    image.append(img)
    plt.imshow(img)
    plt.show()
    txt=[]
    df=pd.read_csv("./project/txt/RFID_100.txt",header=None,sep=' ')
    txt.append(df[0].tolist())
    txt.append(df[2].tolist())
    txt.append(df[4].tolist())
    q = Queue()
    p0 = Process(target=predict_depth, args=(txt,image,q))
    p0.start()
    res=q.get()
    p0.join()
    plt.imshow(res[0])
    plt.show()
    image=[]
    img=cv2.imread("./project/rgb/color188.png")
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img/255
    image.append(img)
    q = Queue()
    p1=Process(target=predict_point, args=(image,q))
    p1.start()
    point=q.get()
    p1.join()
    point=np.array(point)
    point=np.squeeze(point)
    print(point.shape)
    p2list=[]
    p3list=[]
    for i in range(4):
        plt.imshow(point[i])
        plt.show()
        p2=[]
        p3=[]
        p2p3(p2,p3,point[i],res[0])
        p2list.append(p2)
        p3list.append(p3)
    p2list=np.array(p2list,dtype=np.float64)
    p3list=np.array(p3list,dtype=np.float64)
    print(p3list,p3list.shape)
    print(p2list,p2list.shape)
    print(point3d,point3d.shape)
    distCoeffs = None
    print(point3d.shape,p2list.shape,mtx.shape)
    print(type(point3d[0][0]),type(p2list[0][0]),type(mtx[0][0]))
    retval,rvec,tvec  = cv2.solvePnP(p3list,p2list, mtx, distCoeffs,flags=cv2.SOLVEPNP_UPNP)
    print(retval)
    print(rvec)
    print(tvec)
    print(rvec*(180/math.pi))
