import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:144:16, 0:96:16].T.reshape(-1, 2)
#print(objp)
obj_points = []  
img_points = []  

images = glob.glob("./txt/*.png")
for fname in images:
    img = cv2.imread(fname)
    cv2.imshow('img',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
    print(ret)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 6), corners, ret)  
        cv2.imshow('img', img)
        cv2.waitKey(2000)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) 
print("dist:\n", dist)  
print("rvecs:\n", rvecs)  
print("tvecs:\n", tvecs )
s = cv2.FileStorage('./mtx.json', cv2.FileStorage_WRITE)
s.write('mtx', mtx)
s.release()
print("-----------------------------------------------------")