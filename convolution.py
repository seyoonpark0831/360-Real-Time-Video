# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:00:14 2016

@author: psy3061
"""

import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument( required=True,help="path to the first image")
#ap.add_argument( required=True,help="path to the second image")
#args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread('1464562824_3_cylin_frame2.jpg')
imageB = cv2.imread('1464562824_3_cylin_frame3.jpg')
#imageA = imutils.resize(imageA, width=400)
#imageB = imutils.resize(imageB, width=400)

from_px = 0
to_px = 0

max_width = np.size(imageA,1)

compare_value_list = list()

while True:
    if from_px == 0 and to_px < max_width:
        to_px = to_px + 1
    else:
        from_px = from_px + 1
    
    if from_px == max_width:
        break

    #print from_px, to_px

    crop_imageA = imageA[:,from_px:to_px]
    crop_imageB = imageB[:,max_width-to_px:max_width-from_px]
    
    
    
    err = np.sum((crop_imageA.astype('float')-crop_imageB.astype('float'))**2)
    err /= float(crop_imageA.shape[0] * crop_imageA.shape[1])    
    
    compare_value_list.append(err)    
    
    #cv2.imshow("crop1",crop_imageA)
    #cv2.imshow("crop2",crop_imageB)


    #key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #    break

# do a bit of cleanup
#print("[INFO] cleaning up...")
#cv2.destroyAllWindows()

#crop_imageA = imageA[:,1:100]
#crop_imageB = imageB[:,301:400]

#cv2.imshow("crop1",crop_imageA)
#cv2.imshow("crop2",crop_imageB)
#cv2.waitKey(0)

min_err_px = compare_value_list.index(min(compare_value_list))

print min_err_px


fig = plt.figure()
plt.plot(compare_value_list)

plt.show()


"""
if min_err_px > 400:
    to_px = 400
    from_px = min_err_px - 400
else:
    to_px = min_err_px
    from_px = 0
crop_imageA = imageA[:,from_px:to_px]
crop_imageB = imageB[:,max_width-to_px:max_width-from_px]


left_remain_image = imageA[:,0:from_px-1]
left_for_stitch = imageA[:,from_px:to_px]

right_remain_image = imageB[:,max_width-from_px+1:]
right_for_stitch = imageB[:,max_width-to_px:max_width-from_px]



cv2.imshow("crop1",crop_imageA)
cv2.imshow("crop2",crop_imageB)
#cv2.imshow("stitch_resize",resized)

flip_crop_imageA = cv2.flip(crop_imageA,1)
flip_crop_imageB = cv2.flip(crop_imageB,1)

cv2.imwrite('flip_A.png',flip_crop_imageA)
cv2.imwrite('flip_B.png',flip_crop_imageB)


concat_img = np.concatenate((left_remain_image,result,right_remain_image),axis=1)
cv2.imshow("total",concat_img)






cv2.waitKey(0)
"""


