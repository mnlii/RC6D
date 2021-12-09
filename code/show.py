import sys
import os
import cv2
from keras import models 
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
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops.array_ops import rank_internal
from tensorflow.python.ops.gen_array_ops import depth_to_space
import math
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import msvcrt

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
point_init=[1,1,1]
point_init=np.array(point_init)

def init(name1,name2,q):
    img=cv2.imread(name1)
    img=cv2.resize(img,(160,160))
    img=img/255
    img1=[]
    img1.append(img)
    img1=np.array(img1)
    txt=[]
    df=pd.read_csv(name2,header=None,sep=' ')
    txt.append(df[0].tolist())
    txt.append(df[2].tolist())
    txt.append(df[4].tolist())
    txt=np.array(txt)
    txt=np.expand_dims(txt,axis=2)
    txt=txt.reshape(1,3,132)
    model=load_model("./project/result.h5")
    result=model.predict([img1,txt])
    result=np.squeeze(result)
    q.put([result])

def predict_position(name,q):
    img=cv2.imread(name)
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img/255
    img1=[]
    img1.append(img)
    img1=np.array(img1)
    model=load_model("./project/result_V1.h5")
    img1=model.predict(img1)
    img1=np.squeeze(img1)
    img1=img1.reshape(4,160,160)
    q.put([img1])

def predict2d(name,q):
        img=cv2.imread(name)
        img=cv2.resize(img,(160,160))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=img/255
        #plt.imshow(img)
        #plt.show()
        img1=[]
        img1.append(img)
        img1=np.array(img1)
        model=load_model("./project/result_V1.h5")
        img1=model.predict(img1)
        img1=np.squeeze(img1)
        img1=img1.reshape(4,160,160)
        vector2d=[]
        for i in range(4):
            v2d=[]
            img2=img1[i]
            img2=cv2.resize(img1[i],(1920,1080))
            index = np.unravel_index(img2.argmax(), img2.shape)
            print("-----------------index----------------------:",index)
            v2d.append(index[0])
            v2d.append(index[1])
            vector2d.append(v2d)
            #plt.imshow(img2)
            #plt.show()
        vector2d=np.array(vector2d,dtype=np.float64)
        q.put([vector2d])

def init_position(depth,img):
    depth=cv2.resize(depth,(1920,1080))
    vector2d=[]
    vector3d=[]
    for i in range(4):
        v2d=[]
        v3d=[]
        img1=img[i]
        img1=cv2.resize(img[i],(1920,1080))
        index = np.unravel_index(img1.argmax(), img1.shape)
        v2d.append(index[0])
        v2d.append(index[1])
        v3d.append(index[0])
        v3d.append(index[1])
        v3d.append(depth[index[0]][index[1]]*255*255)
        vector2d.append(v2d)
        vector3d.append(v3d)
    vector2d=np.array(vector2d,dtype=np.float64)
    vector3d=np.array(vector3d,dtype=np.float64)
    print(vector3d,vector3d.shape)
    print(vector2d,vector2d.shape)
    s = cv2.FileStorage()
    s.open('./mtx.json', cv2.FileStorage_READ)
    mtx = s.getNode('mtx').mat()
    distCoeffs = None
    retval,rvec,tvec  = cv2.solvePnP(vector3d,vector2d, mtx, distCoeffs,flags=cv2.SOLVEPNP_UPNP)
    print(retval)
    print(rvec)
    print(tvec)
    print(rvec*(180/math.pi))
    s = cv2.FileStorage('./opencv_temp.json', cv2.FileStorage_WRITE)
    s.write('vector p3', vector3d)
    s.release()

def pnp(a,b,c):
    distCoeffs = None
    print(a,b,c)
    retval,rvec,tvec  = cv2.solvePnP(a,b,c, distCoeffs,flags=cv2.SOLVEPNP_UPNP)
    rvec1=rvec*(180/math.pi)
    return rvec,tvec,rvec1

def on_key_press(event):
    print(event.key)
    exit(0)

def draw(name,point,fname,fpoint):
    img=cv2.imread(name)
    #img=cv2.resize(img,(160,160))
    for i in point:
        cv2.circle(img, (int(i[1]),int(i[0])), 1, (0,255,0),4)
    #img=cv2.resize(img,(160,160))
    img1=cv2.imread(fname)
    #img1=cv2.resize(img1,(160,160))
    for i in fpoint:
        cv2.circle(img1, (int(i[1]),int(i[0])), 1, (0,255,0),4)
    #img1=cv2.resize(img1,(160,160))
    imgs = np.hstack([img,img1])
    print(imgs.shape)
    for i ,ps in enumerate(point):
        cv2.line(imgs, (int(point[i][1]),int(point[i][0])), (int(fpoint[i][1])+1920,int(fpoint[i][0])), (0,255,0), 3, 4)
        #print(int(point[i][1]),int(point[i][0]), int(fpoint[i][1])+1920,int(fpoint[i][0]))
    imgs=cv2.resize(imgs,(1040,520))
    # fig = plt.figure()
    # fig.canvas.mpl_connect("button_press_event", on_key_press) 
    # fig.imshow(imgs)
    # plt.axis("off")
    # plt.pause(4)
    cv2.imshow("imgs",imgs)
    if(cv2.waitKey(4)==27):
        cv2.imwrite("result.png",imgs)
        return 1
    return 0

if __name__=='__main__':
    is_init=True
    if(is_init==True):
        imagename="./project/rgb/color0.png"
        rfidname="./project/txt/RFID_0.txt"
        q = Queue()
        p0 = Process(target=init, args=(imagename,rfidname,q))
        p0.start()
        res=q.get()
        p0.join()
        plt.imshow(res[0])
        plt.show()
        q = Queue()
        p1=Process(target=predict_position, args=(imagename,q))
        p1.start()
        point=q.get()
        p1.join()
        point=np.array(point)
        plt.imshow(point[0][0])
        plt.show()
        plt.imshow(point[0][1])
        plt.show()
        plt.imshow(point[0][2])
        plt.show()
        plt.imshow(point[0][3])
        plt.show()
        init_position(res[0],point[0])
    s = cv2.FileStorage()
    s.open('./opencv_temp.json', cv2.FileStorage_READ)
    point3d = s.getNode('vector p3').mat()
    print(point3d,point3d.shape)
    s = cv2.FileStorage()
    s.open('./mtx.json', cv2.FileStorage_READ)
    mtx = s.getNode('mtx').mat()
    print(mtx)
    fimagename="./project/rgb/color0.png"
    filename=os.listdir("./project/rgb")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in filename:
        imagename="./project/rgb/"+i
        q = Queue()
        p0 = Process(target=predict2d, args=(imagename,q))
        p0.start()
        res=q.get()
        print(res[0],res[0].shape)
        q = Queue()
        r,t,d=pnp(point3d,res[0],mtx)
        print(r)
        print(t)
        print(d)
        turn=draw(fimagename,point3d,imagename,res[0])
        rm=cv2.Rodrigues(r)
        t=np.squeeze(t)
        point=point_init
        point_init=np.dot(point_init,rm[0])+t
        ax.scatter3D([point[0],point_init[0]],[point[1],point_init[1]], [point[2],point_init[2]],cmap='Greens')
        ax.plot([point[0],point_init[0]],[point[1],point_init[1]], [point[2],point_init[2]], 'gray')
        plt.pause(4)
        if(turn==1):
            plt.savefig("tracker.png")
            plt.close()
            break

