from operator import index
import numpy as np
import os
import pandas as pd
from pandas.core.accessor import PandasDelegate
import matplotlib.pyplot as plt
from pandas.core.indexing import IndexSlice
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.python.ops.gen_lookup_ops import lookup_table_export
from multiprocessing import Process,Queue
import math
# model1=keras.models.load_model("./project/resultpoint.h5")
# model2=keras.models.load_model("./project/result.h5")
def load_txt(name1):
    name="./project/txt/RFID_"+name1[5:-4]+".txt"
    txt=[]
    df=pd.read_csv(name,header=None,sep=' ')
    txt.append(df[0].tolist())
    txt.append(df[2].tolist())
    txt.append(df[4].tolist())
    txt=np.array(txt)
    txt=np.expand_dims(txt,axis=2)
    txt=txt.reshape(1,3,132)
    return txt
def load_image(name):
    img=cv2.imread("./project/rgb/"+name)
    depth=cv2.imread("./project/depth/depth"+name[5:-4]+"_trans.png",-1)
    labels=[]
    indexs=[]
    label2d=[]
    for i in range(4):
        label=cv2.imread("./project/label/"+name[5:-4]+"_"+str(i)+".png")
        label=cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
        index = np.unravel_index(label.argmax(), label.shape)
        v3d=[]
        #print(index)
        v3d.append(index[0])
        v3d.append(index[1])
        v3d.append(depth[index[0]][index[1]])
        indexs.append(v3d)
        v2d=[]
        #print(index)
        v2d.append(index[0])
        v2d.append(index[1])
        labels.append(label)
        label2d.append(v2d)
    img=cv2.resize(img,(160,160))
    label=cv2.resize(label,(160,160))
    depth=cv2.resize(depth,(160,160))
    return img,depth,labels,indexs,label2d
def point_detection(rgb,q):
    img1=[]
    rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    rgb=rgb/255
    img1.append(rgb)
    img1=np.array(img1)
    model=keras.models.load_model("./project/result_V1.h5")
    img1=model.predict(img1)
    img1=np.squeeze(img1)
    img1=img1.reshape(4,160,160)
    # plt.imshow(img1[0])
    # plt.show()
    point=[]
    for i in range(4):
        img=img1[i]
        img=cv2.resize(img,(1920,1080))
        index = np.unravel_index(img.argmax(), img.shape)
        v2d=[]
        v2d.append(index[0])
        v2d.append(index[1])
        point.append(v2d)
    q.put(point)
def depth_g(rgb,res,txt,q):
    #rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    rgb=rgb/255
    img1=[]
    img1.append(rgb)
    img1=np.array(img1)
    #print(img1.shape)
    txt=np.expand_dims(txt,axis=2)
    txt=txt.reshape(1,3,132)
    model=keras.models.load_model("./project/result.h5")
    result=model.predict([img1,txt])
    result=np.squeeze(result)
    point=[]
    for i in range(4):
        p=res[i]
        img=cv2.resize(result,(1920,1080))
        index = img[p[0]][p[1]]
        v2d=[]
        v2d.append(p[0])
        v2d.append(p[1])
        v2d.append(index*256*256)
        point.append(v2d)
    q.put(point)
if __name__ == '__main__':
    filename=os.listdir("./project/rgb/")
    for k in range(7):
        for i in filename:
            txt=load_txt(i)
            rgb,depth,labels,Indexs,l2d=load_image(i)
            q = Queue()
            p0 = Process(target=point_detection, args=(rgb,q))
            p0.start()
            res=q.get()
            p0.join()
            print(res)
            print(Indexs)
            q = Queue()
            p0 = Process(target=depth_g, args=(rgb,res,txt,q))
            p0.start()
            result=q.get()
            p0.join()
            print(Indexs)
            Indexs=np.array(Indexs,dtype=np.float64)
            result=np.array(result,dtype=np.float64)
            res=np.array(res,dtype=np.float64)
            l2d=np.array(l2d,dtype=np.float64)
            print(result)
            s = cv2.FileStorage()
            s.open('./mtx.json', cv2.FileStorage_READ)
            mtx = s.getNode('mtx').mat()
            distCoeffs = None
            retval,rvec,tvec  = cv2.solvePnP(Indexs,l2d,mtx, distCoeffs,flags=cv2.SOLVEPNP_UPNP)
            rvec1=rvec*(180/math.pi)
            retval_p,rvec_p,tvec_p  = cv2.solvePnP(result,res,mtx, distCoeffs,flags=cv2.SOLVEPNP_UPNP)
            rvec_p_1=rvec_p*(180/math.pi)
            fo = open("Disinfection_error.txt", "a+")
            for i in range(3):
                r=abs(rvec1[i]-rvec_p_1[i])
                t=abs(tvec[i]-tvec_p[i])
                while(t>200):
                    t=t/10
                while(r>80):
                    r=r/10
                fo.write(str(r)+" "+str(t))
                print(r,t)
            fo.write('\n')
            fo.close()
            # rgb=cv2.resize(rgb,(1920,1080))
            # rgb=cv2.circle(rgb, (int(p_2d[0][1]),int(p_2d[0][0])), 10, (0,0,0),10)
            # print((int(p_2d[0][0]),int(p_2d[0][1])))
            # plt.imshow(rgb)
            # plt.show()
            # plt.subplot(2,2,1)
            # plt.imshow(rgb)
            # plt.subplot(2,2,2)
            # plt.imshow(depth)
            # plt.subplot(2,2,3)
            # plt.imshow(labels[0])
            # plt.subplot(2,2,4)
            # plt.imshow(labels[1])
            # plt.show()
    



