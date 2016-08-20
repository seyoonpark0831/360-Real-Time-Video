# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:32:14 2016

@author: psy3061
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:41:13 2016

@author: psy3061
"""

import cv2
import imutils
import numpy as np

from timeit import default_timer as timer





def makeOriMap(y,x):
    return np.dstack((x,y)).astype(np.int16)


def doProjection(img):
    out_img = np.multiply(img,0)
    
    xdim = img.shape[1]
    ydim = img.shape[0]
    
    xc = xdim / 2
    yc = ydim / 2
    
    
    f = np.float(950)
    k1 = 0.1
    k2 = 0.1
    
    
    
    theta = lambda y,x : (x - xc) / f
    h = lambda y,x : (y - yc) / f
    
    xcap = lambda y,x : np.sin(theta(y,x))
    ycap = lambda y,x : h(y,x)
    zcap = lambda y,x : np.cos(theta(y,x))
    xn = lambda y,x : xcap(y,x) / zcap(y,x)
    yn = lambda y,x : ycap(y,x) / zcap(y,x)
    r = lambda y,x : xn(y,x)**2 + yn(y,x)**2
    
    xd = lambda y,x : xn(y,x) * (1 + k1 * r(y,x) + k2 * (r(y,x)**2))
    yd = lambda y,x : yn(y,x) * (1 + k1 * r(y,x) + k2 * (r(y,x)**2))
    
    toXimg = lambda y,x : (np.floor(f * xd(y,x) + xc)).astype(int)
    toYimg = lambda y,x : (np.floor(f * yd(y,x) + yc)).astype(int)


    def makeMap(y, x):
        #print x, y    
        
        ximg = toXimg(y,x)
        yimg = toYimg(y,x)
        return np.dstack((ximg, yimg)).astype(np.int16)

    map_ori_xy = np.fromfunction(makeOriMap, img.shape[:2], dtype=np.int16)

    map_xy = np.fromfunction(makeMap, img.shape[:2], dtype=np.int16)
    
    #start = timer()
    img_mapped = cv2.remap(img, (map_xy), None, False)
    #end = timer()
    #print(end - start)    
    
    
    #cv2.imwrite("right_proj.png",img_mapped)
    
    
    
    ## Find Contour Region
    gray = cv2.cvtColor(img_mapped,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    
    ## Crop Image
    #start = timer()
    crop = img_mapped[y:y+h,x:x+w]
    #end = timer()
    #print(end - start)  
    
    #cv2.imwrite('right_proj_crop.png',crop)    
    
    return crop
    
    
    
    
    






