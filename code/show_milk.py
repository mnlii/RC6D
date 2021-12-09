import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import polyint
import datetime

filename=os.listdir("./zkx/txt/rgb/")
fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')  
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

def drawpoint(imgname):
    labelname=imgname[0:-4]
    img=cv2.imread("./zkx/txt/rgb/"+imgname,-1)
    point=[]
    for i in range(4):
        label=cv2.imread("./zkx/txt/label/"+labelname+"_"+str(i)+".png",-1)
        index = np.unravel_index(label.argmax(), label.shape)
        point.append(index)
        cv2.circle(img, (int(index[1]),int(index[0])), 10, (0,255,0),4)
    #plt.imshow(img)
    #plt.show()
    return point,img

def match(img1,img2,point,fpoint):
    imgs = np.hstack([img1,img2])
    for i ,ps in enumerate(point):
        cv2.line(imgs, (int(point[i][1]),int(point[i][0])), (int(fpoint[i][1])+1920,int(fpoint[i][0])), (0,255,0), 3, 4)
    imgs=cv2.resize(imgs,(640,480))
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    cv2.imshow("imgs",imgs)
    out.write(imgs)
    if(cv2.waitKey(4)==27):
        return 1
    return 0

if __name__ == '__main__':
    nums=0
    starttime = datetime.datetime.now()
    name="0.png"
    p0,img0=drawpoint(name)
    for i in filename:
        p1,img1=drawpoint(i)
        flag=match(img0,img1,p0,p1)
        if(flag==1):
            break
        nums=nums+1
        if(nums==40):
            break
    out.release()
    cv2.destroyAllWindows()
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    fo = open("time.txt", "w")
    fo.write(str((endtime - starttime).seconds))
    fo.close()
    # cap = cv2.VideoCapture('output.avi')
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('frame',gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

