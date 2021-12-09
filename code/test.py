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
    #plt.imshow(result)
    #plt.show()

#K.clear_session()
#cuda.select_device(0)
#cuda.close()
def predict_point(test,q):
    model=load_model("./project/result_u.h5")
    test=np.array(test)
    print(test.shape)
    img1=model.predict(test)
    img1=np.squeeze(img1)
    img1=img1.reshape(4,160,160)
    q.put([img1])

if __name__=='__main__':
    imgd=cv2.imread("./project/depth/depth0_trans.png",-1)
    img=cv2.imread("./project/rgb/color0.png")
    image=[]
    imgp=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(160,160))
    imgp=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img/255
    imgp=imgp/255
    image.append(img)
    plt.imshow(img)
    plt.show()
    txt=[]
    df=pd.read_csv("./project/txt/RFID_0.txt",header=None,sep=' ')
    txt.append(df[0].tolist())
    txt.append(df[2].tolist())
    txt.append(df[4].tolist())
    imgd=cv2.resize(imgd,(160,160))
    print(imgd.shape)
    imgd=imgd/(255*255)
    plt.imshow(imgd)
    plt.show()
    q = Queue()
    p0 = Process(target=predict_depth, args=(txt,image,q))
    p0.start()
    res=q.get()
    p0.join()
    plt.imshow(res[0])
    plt.show()
    res1=res[0]*255
    cv2.imwrite("model.png",res1)
    image.clear()
    image.append(imgp)
    q = Queue()
    p1=Process(target=predict_point, args=(image,q))
    p1.start()
    point=q.get()
    p1.join()
    point=np.array(point)
    print(point.shape)
    point=np.squeeze(point)
    p21=[]
    p22=[]
    p23=[]
    p24=[]
    p31=[]
    p32=[]
    p33=[]
    p34=[]
    p2p3(p21,p31,point[0],res[0])
    p2p3(p22,p32,point[1],res[0])
    p2p3(p23,p33,point[2],res[0])
    p2p3(p24,p34,point[3],res[0])
    for i in range(4):
        plt.imshow(point[i])
    print(p21,' ',p31)
    print(p22,' ',p32)
    print(p23,' ',p33)
    print(p24,' ',p34)
    p3list=[]
    p3list.append(p31)
    p3list.append(p32)
    p3list.append(p33)
    p3list.append(p34)
    p3list=np.array(p3list)
    s = cv2.FileStorage('./opencv_temp.json', cv2.FileStorage_WRITE)
    s.write('vector p3', p3list)
    s.release()