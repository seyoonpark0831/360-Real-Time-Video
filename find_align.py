
import glob

import stitch
import numpy as np
import cv2
import imutils
from timeit import default_timer as timer




img1 = cv2.imread('1464562824_3_cylin_frame4.jpg')
img2 = cv2.imread('1464562824_3_cylin_frame5.jpg')


print img1.shape[1]


M =  stitch.get_sift_homography(img1,img2)
x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(img2,img1,M)



# Create output array after affine transformation 
transform_dist = [-x_min,-y_min]
transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

stitch_img = stitch.get_stitched_image(img2,img1,M)

# Warp images to get the resulting image
result_img = cv2.warpPerspective(img1, transform_array.dot(M),(x_max-x_min, y_max-y_min))


print transform_array.dot(M)



original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])


print transform_array.dot(M) - original_trans

print sum(sum(abs(transform_array.dot(M) - original_trans)))


result_img2 = cv2.warpPerspective(img1, np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]]),(x_max-x_min, y_max-y_min))




cv2.imshow("SIFT Stitch", result_img)
cv2.imshow("SIFT Stitch2", result_img2)
cv2.waitKey(0)
start = 0
end = 0

for x in range(result_img.shape[1]):

	if start == 0 and np.sum(result_img[:,x]) != 0:
		start = x
	elif start != 0 and np.sum(result_img[:,x]) == 0:
		end = x
		break



print end-start









#print x_min,x_max, y_min,y_max

#print x_max-x_min, y_max-y_min
#print transform_array.dot(M)

#cv2.imshow("SIFT Stitch", result_img)
#cv2.imshow("SIFT Stitch2", stitch_img)


#cv2.waitKey(0)
