import sys
from PyQt4 import QtCore, QtGui

from ex_ui2 import Ui_MainWindow


import glob

import stitch
import numpy as np
import cv2
import imutils
from timeit import default_timer as timer
import threading

import seam as seamfunc

from multiprocessing import Process,Queue

import pickle
from imutils.video import WebcamVideoStream





class MyGui(QtGui.QMainWindow):


	# Flags
	showCameraFlag = False
	featureFlag = False
	alignFlag01 = False
	alignFlag12 = False
	alignFlag23 = False
	alignFlag34 = False
	alignFlag45 = False

	videoStitchFlag = False

	# Video Read
	cam_0 = WebcamVideoStream(src=0)
	cam_0.stream.set(3,1280)
	cam_0.stream.set(4,720)
	cam_0.stream.set(5,30)
	cap_0 = cam_0.start()

	cam_1 = WebcamVideoStream(src=1)
	cam_1.stream.set(3,1280)
	cam_1.stream.set(4,720)
	cam_1.stream.set(5,30)
	cap_1 = cam_1.start()

	cam_2 = WebcamVideoStream(src=2)
	cam_2.stream.set(3,1280)
	cam_2.stream.set(4,720)
	cam_2.stream.set(5,30)
	cap_2 = cam_2.start()

	cam_3 = WebcamVideoStream(src=3)
	cam_3.stream.set(3,1280)
	cam_3.stream.set(4,720)
	cam_3.stream.set(5,30)
	cap_3 = cam_3.start()

	cam_4 = WebcamVideoStream(src=4)
	cam_4.stream.set(3,1280)
	cam_4.stream.set(4,720)
	cam_4.stream.set(5,30)
	cap_4 = cam_4.start()

	cam_5 = WebcamVideoStream(src=5)
	cam_5.stream.set(3,1280)
	cam_5.stream.set(4,720)
	cam_5.stream.set(5,30)	
	cap_5 = cam_5.start()


	num = 0

	map_cylin_xy = 0
	cylinX = 0
	cylinY = 0
	cylinW = 0
	cylinH = 0


	images = list()
	print images

	# Basic Variables
	h_matrix_list = list()
	min_max_list = list()

	transform_matrix_list = list()
	final_transform_matrix_list = list()
	# Get Total Width & Total Height From Last Min Max Values
	total_width = 0
	total_height = 0

	cut_position_list = list()

	stitch_width_list = [2144,3072,4136,5132,6016]


	combined_mask_list = list()

	trans_images = list()
	seam_list = list()

	
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self, parent)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		#self.ui.align_label_01.setText("TESTTEST")

		self.map_cylin_xy, self.cylinX, self.cylinY, self.cylinW, self.cylinH = self.cylinMapping()


		self.ui.take_shot_btn.clicked.connect(self.take_shot_btn_clicked)
		self.ui.take_video_btn.clicked.connect(self.take_video_btn_clicked)

		self.ui.show_camera_btn.clicked.connect(self.show_camera_btn_clicked)

		self.ui.feature_check_stop_btn.clicked.connect(self.feature_check_stop_btn_clicked)

		self.ui.align_btn_01.clicked.connect(self.align_btn_01_clicked)
		self.ui.align_btn_12.clicked.connect(self.align_btn_12_clicked)
		self.ui.align_btn_23.clicked.connect(self.align_btn_23_clicked)
		self.ui.align_btn_34.clicked.connect(self.align_btn_34_clicked)
		self.ui.align_btn_45.clicked.connect(self.align_btn_45_clicked)

		self.ui.align_stop_btn_01.clicked.connect(self.align_stop_btn_01_clicked)
		self.ui.align_stop_btn_12.clicked.connect(self.align_stop_btn_12_clicked)
		self.ui.align_stop_btn_23.clicked.connect(self.align_stop_btn_23_clicked)
		self.ui.align_stop_btn_34.clicked.connect(self.align_stop_btn_34_clicked)
		self.ui.align_stop_btn_45.clicked.connect(self.align_stop_btn_45_clicked)

		self.ui.img_stitch_btn.clicked.connect(self.img_stitch_btn_clicked)

		self.ui.save_params_btn.clicked.connect(self.save_params_btn_clicked)
		self.ui.load_params_btn.clicked.connect(self.load_params_btn_clicked)

		self.ui.create_mask_btn.clicked.connect(self.create_mask_btn_clicked)

		self.ui.img_stitch_without_seam_btn.clicked.connect(self.img_stitch_without_seam_btn_clicked)
		self.ui.img_stitch_with_seam_btn.clicked.connect(self.img_stitch_with_seam_btn_clicked)
		self.ui.video_stitch_without_seam_btn.clicked.connect(self.video_stitch_without_seam_btn_clicked)
		self.ui.video_stitch_with_seam_btn.clicked.connect(self.video_stitch_with_seam_btn_clicked)




	def take_shot_btn_clicked(self, parent=None):
		QtGui.QApplication.processEvents()
		frame_0 = self.cap_0.read()
		frame_1 = self.cap_1.read()
		frame_2 = self.cap_2.read()
		frame_3 = self.cap_3.read()
		frame_4 = self.cap_4.read()
		frame_5 = self.cap_5.read()

		
		self.recaptureFrames()

		time = int(timer())

        	cv2.imwrite('shots/%d_%d_ori_frame0.jpg'%(time,self.num),frame_0)
		cv2.imwrite('shots/%d_%d_ori_frame1.jpg'%(time,self.num),frame_1)
		cv2.imwrite('shots/%d_%d_ori_frame2.jpg'%(time,self.num),frame_2)
		cv2.imwrite('shots/%d_%d_ori_frame3.jpg'%(time,self.num),frame_3)
		cv2.imwrite('shots/%d_%d_ori_frame4.jpg'%(time,self.num),frame_4)
		cv2.imwrite('shots/%d_%d_ori_frame5.jpg'%(time,self.num),frame_5)

		cv2.imwrite('shots/%d_%d_cylin_frame0.jpg'%(time,self.num),self.images[0])
		cv2.imwrite('shots/%d_%d_cylin_frame1.jpg'%(time,self.num),self.images[1])
		cv2.imwrite('shots/%d_%d_cylin_frame2.jpg'%(time,self.num),self.images[2])
		cv2.imwrite('shots/%d_%d_cylin_frame3.jpg'%(time,self.num),self.images[3])
		cv2.imwrite('shots/%d_%d_cylin_frame4.jpg'%(time,self.num),self.images[4])
		cv2.imwrite('shots/%d_%d_cylin_frame5.jpg'%(time,self.num),self.images[5])
		self.num = self.num + 1


	def take_video_btn_clicked(self, parent=None):
		now = timer()
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out0 = cv2.VideoWriter("videos/video_%d_0.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		out1 = cv2.VideoWriter("videos/video_%d_1.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		out2 = cv2.VideoWriter("videos/video_%d_2.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		out3 = cv2.VideoWriter("videos/video_%d_3.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		out4 = cv2.VideoWriter("videos/video_%d_4.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		out5 = cv2.VideoWriter("videos/video_%d_5.avi"%(now), fourcc, 13.0, ((self.cylinX+self.cylinW-10-(self.cylinX+10)),self.cylinH))
		while True:
			QtGui.QApplication.processEvents()
			start = timer()

			self.recaptureFrames()
			out0.write(self.images[0])
			out1.write(self.images[1])
			out2.write(self.images[2])
			out3.write(self.images[3])
			out4.write(self.images[4])
			out5.write(self.images[5])
			end = timer()
			self.ui.feature_label.setText("Time: %.5f"%(end-start)+"   \nFPS : %.1f"%(1/(end-start)))



	def show_camera_btn_clicked(self, parent=None):
		print "Show Camera Btn Press"

		self.showCameraFlag = True

		while(self.showCameraFlag):
			QtGui.QApplication.processEvents()
			frame_0 = self.cap_0.read()
			frame_1 = self.cap_1.read()
			frame_2 = self.cap_2.read()
			frame_3 = self.cap_3.read()
			frame_4 = self.cap_4.read()
			frame_5 = self.cap_5.read()


			#if ret_0==True:

			img_mapped_0 = cv2.remap(frame_0, (self.map_cylin_xy), None, False)
			img_mapped_1 = cv2.remap(frame_1, (self.map_cylin_xy), None, False)
			img_mapped_2 = cv2.remap(frame_2, (self.map_cylin_xy), None, False)
			img_mapped_3 = cv2.remap(frame_3, (self.map_cylin_xy), None, False)
			img_mapped_4 = cv2.remap(frame_4, (self.map_cylin_xy), None, False)
			img_mapped_5 = cv2.remap(frame_5, (self.map_cylin_xy), None, False)


			crop_0 = img_mapped_0[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_1 = img_mapped_1[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_2 = img_mapped_2[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_3 = img_mapped_3[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_4 = img_mapped_4[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_5 = img_mapped_5[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]

			cv2.imshow("Feature",imutils.resize(np.concatenate((np.concatenate((crop_0, crop_1, crop_2), axis=1),np.concatenate((crop_2, crop_3, crop_4), axis=1),np.concatenate((crop_3, crop_4, crop_5), axis=1)), axis=0),width=1000))


			cv2.waitKey(1)
				
			#else:
			#	break







	def feature_check_stop_btn_clicked(self, parent=None):
		print "Feature Check Stop Btn Press"
		self.showCameraFlag = False
		

	def align_btn_01_clicked(self, parent=None):
		print "Align Btn 01 Press"

		self.alignFlag01 = True
		frame_0 = self.cap_0.read()
		frame_1 = self.cap_1.read()

		img_mapped_0 = cv2.remap(frame_0, (self.map_cylin_xy), None, False)
		img_mapped_1 = cv2.remap(frame_1, (self.map_cylin_xy), None, False)


		crop_0 = img_mapped_0[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_1 = img_mapped_1[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]

		img1 = crop_0
		img2 = crop_1



		# Get width and height of input images	
		w1,h1 = img1.shape[:2]
		w2,h2 = img2.shape[:2]

		# Get the canvas dimesions
		img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

		optimal_transform_matrix_change_value = 100
		optimal_transform_matrix = None
		optimal_x_max = 0
		optimal_x_min = 0
		optimal_y_max = 0
		optimal_y_min = 0	


		min_error_width = 500
		min_error_height = 100
		video_height = 717

		while(self.alignFlag01):
			QtGui.QApplication.processEvents()
			frame_0 = self.cap_0.read()
			frame_1 = self.cap_1.read()

			#if ret_0==True:

			img_mapped_0 = cv2.remap(frame_0, (self.map_cylin_xy), None, False)
			img_mapped_1 = cv2.remap(frame_1, (self.map_cylin_xy), None, False)


			crop_0 = img_mapped_0[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
			crop_1 = img_mapped_1[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]

			img1 = crop_0
			img2 = crop_1


			feature_str = ""

			MIN_MATCH_COUNT = 10

			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create() 

			# find the keypoints and descriptors with SIFT
			k1, d1 = sift.detectAndCompute(img1,None)
			k2, d2 = sift.detectAndCompute(img2,None)




			# Bruteforce matcher on the descriptors
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(d1,d2, k=2)

			# Make sure that the matches are good
			verify_ratio = 0.8 # Source: stackoverflow
			verified_matches = []
			for m1,m2 in matches:
				# Add to array only if it's a good match
				if m1.distance < 0.8 * m2.distance:
					verified_matches.append(m1)

			# Mimnum number of matches

			if len(verified_matches) > MIN_MATCH_COUNT:
	
				# Array to store matching points
				img1_pts = []
				img2_pts = []

				# Add matching points to array
				for match in verified_matches:
					img1_pts.append(k1[match.queryIdx].pt)
					img2_pts.append(k2[match.trainIdx].pt)
				img1_pts = np.float32(img1_pts).reshape(-1,1,2)
				img2_pts = np.float32(img2_pts).reshape(-1,1,2)
	
				# Compute homography matrix
				M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
				feature_str = "Good :"+str(len(verified_matches))
			else:
				#print 'Error: Not enough matches'
				feature_str = "Not Good :"+str(len(verified_matches))+"/"+str(MIN_MATCH_COUNT)


			#print feature_str



			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = mask.ravel().tolist(), # draw only inliers
					   flags = 2)

			feature_match_img = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)



			# Get relative perspective of second image
			img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

			# Resulting dimensions
			result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)



			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

			# Create output array after affine transformation 
			transform_dist = [-x_min,-y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

			original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])
	
			transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))

			


			# Resulting dimensions
			result_dims = np.concatenate( (img2_dims, img1_dims), axis = 0)

			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
			

			temp_stitch_img = stitch.get_stitched_right_image(img2,img1,M)

			if optimal_transform_matrix == None:
				print optimal_transform_matrix
				optimal_transform_matrix = M
				optimal_x_max = x_max
				optimal_x_min = x_min
				optimal_y_max = y_max
				optimal_y_min = y_min

			print str(temp_stitch_img.shape[1]) + " " + str(self.stitch_width_list[0])+ " " + str(temp_stitch_img.shape[0]) + " " + str(video_height)


			if abs(temp_stitch_img.shape[1] - self.stitch_width_list[0]) < 100 and min_error_width > abs(temp_stitch_img.shape[1] - self.stitch_width_list[0]):
				if abs(temp_stitch_img.shape[0] - video_height) < 10 and min_error_height > abs(temp_stitch_img.shape[0] - video_height): 
					print "New Min Value : "+str(transform_matrix_change_value)
					optimal_transform_matrix_change_value = transform_matrix_change_value

					min_error_width = abs(temp_stitch_img.shape[1] - self.stitch_width_list[0])
					min_error_height = abs(temp_stitch_img.shape[0] - video_height)
					optimal_transform_matrix = M
					optimal_x_max = x_max
					optimal_x_min = x_min
					optimal_y_max = y_max
					optimal_y_min = y_min

					if len(self.h_matrix_list) == 0:
						print "01 H Matrix List Append : 0"
						self.h_matrix_list.append(M)
						self.min_max_list.append([x_min, x_max, y_min, y_max])
					else:
						print "01 H Matrix List Value Changed : 0"
						self.h_matrix_list[0] = M
						self.min_max_list[0] = [x_min, x_max, y_min, y_max]


			# Create output array after affine transformation 
			transform_dist = [-optimal_x_min,-optimal_y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

			# Warp images to get the resulting image
			stitch_img = cv2.warpPerspective(img1, transform_array.dot(optimal_transform_matrix),(optimal_x_max-optimal_x_min, optimal_y_max-optimal_y_min))
			stitch_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img2



			


			cv2.imshow("Feature_0_1",imutils.resize(feature_match_img,width=700))
			cv2.imshow("Stitch_0_1",imutils.resize(stitch_img,width=700))
			cv2.waitKey(1)

			self.ui.align_label_01.setText(feature_str+"\n"+str(transform_matrix_change_value)+"\n"+str(temp_stitch_img.shape[1]))
				
			#else:
				#break
			



	def align_btn_12_clicked(self, parent=None):
		print "Align Btn 12 Press"
		self.alignFlag12 = True

		print len(self.h_matrix_list)
		self.recaptureFrames()
		prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])



		img1 = prev_right_img
		img2 = self.images[2]

		min_error_width = 500
		min_error_height = 100
		video_height = 717

		# Get width and height of input images	
		w1,h1 = img1.shape[:2]
		w2,h2 = img2.shape[:2]

		# Get the canvas dimesions
		img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

		optimal_transform_matrix_change_value = 100
		optimal_transform_matrix = None
		optimal_x_max = 0
		optimal_x_min = 0
		optimal_y_max = 0
		optimal_y_min = 0	


		while(self.alignFlag12):
			QtGui.QApplication.processEvents()
			self.recaptureFrames()
			prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])



			img1 = prev_right_img
			img2 = self.images[2]


			feature_str = ""

			MIN_MATCH_COUNT = 10

			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create() 

			# find the keypoints and descriptors with SIFT
			k1, d1 = sift.detectAndCompute(img1,None)
			k2, d2 = sift.detectAndCompute(img2,None)




			# Bruteforce matcher on the descriptors
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(d1,d2, k=2)

			# Make sure that the matches are good
			verify_ratio = 0.8 # Source: stackoverflow
			verified_matches = []
			for m1,m2 in matches:
				# Add to array only if it's a good match
				if m1.distance < 0.8 * m2.distance:
					verified_matches.append(m1)

			# Mimnum number of matches

			if len(verified_matches) > MIN_MATCH_COUNT:
	
				# Array to store matching points
				img1_pts = []
				img2_pts = []

				# Add matching points to array
				for match in verified_matches:
					img1_pts.append(k1[match.queryIdx].pt)
					img2_pts.append(k2[match.trainIdx].pt)
				img1_pts = np.float32(img1_pts).reshape(-1,1,2)
				img2_pts = np.float32(img2_pts).reshape(-1,1,2)
	
				# Compute homography matrix
				M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
				feature_str = "Good :"+str(len(verified_matches))
			else:
				#print 'Error: Not enough matches'
				feature_str = "Not Good :"+str(len(verified_matches))+"/"+str(MIN_MATCH_COUNT)


			#print feature_str



			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = mask.ravel().tolist(), # draw only inliers
					   flags = 2)

			feature_match_img = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)



			# Get relative perspective of second image
			img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

			# Resulting dimensions
			result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)



			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

			# Create output array after affine transformation 
			transform_dist = [-x_min,-y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

			original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])
	
			transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))

			


			# Resulting dimensions
			result_dims = np.concatenate( (img2_dims, img1_dims), axis = 0)

			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
			

			temp_stitch_img = stitch.get_stitched_right_image(img2,img1,M)

			if optimal_transform_matrix == None:
				print optimal_transform_matrix
				optimal_transform_matrix = M
				optimal_x_max = x_max
				optimal_x_min = x_min
				optimal_y_max = y_max
				optimal_y_min = y_min

			print str(temp_stitch_img.shape[1]) + " " + str(self.stitch_width_list[1])+ " " + str(temp_stitch_img.shape[0]) + " " + str(video_height)

			if abs(temp_stitch_img.shape[1] - self.stitch_width_list[1]) < 100 and min_error_width > abs(temp_stitch_img.shape[1] - self.stitch_width_list[1]):
				#if abs(temp_stitch_img.shape[0] - video_height) < 10 and min_error_height > abs(temp_stitch_img.shape[0] - video_height): 
					print "12 New Min Value : "+str(transform_matrix_change_value)
					optimal_transform_matrix_change_value = transform_matrix_change_value

					min_error_width = abs(temp_stitch_img.shape[1] - self.stitch_width_list[0])
					min_error_height = abs(temp_stitch_img.shape[0] - video_height)

					optimal_transform_matrix = M
					optimal_x_max = x_max
					optimal_x_min = x_min
					optimal_y_max = y_max
					optimal_y_min = y_min

					if len(self.h_matrix_list) == 1:
						print "12 H Matrix List Append : 1"
						self.h_matrix_list.append(M)
						self.min_max_list.append([x_min, x_max, y_min, y_max])
					else:
						print "12 H Matrix List Value Changed : 1"
						self.h_matrix_list[1] = M
						self.min_max_list[1] = [x_min, x_max, y_min, y_max]



			# Create output array after affine transformation 
			transform_dist = [-optimal_x_min,-optimal_y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

			# Warp images to get the resulting image
			#stitch_img = cv2.warpPerspective(img1, transform_array.dot(optimal_transform_matrix),(optimal_x_max-optimal_x_min, optimal_y_max-optimal_y_min))
			#stitch_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img2
			stitch_img = stitch.get_stitched_image(img2,img1,optimal_transform_matrix)


			


			cv2.imshow("Feature_0_1",imutils.resize(feature_match_img,width=700))
			cv2.imshow("Stitch_0_1",imutils.resize(stitch_img,width=700))
			cv2.waitKey(1)

			self.ui.align_label_12.setText(feature_str+"\n"+str(transform_matrix_change_value)+"\n"+str(temp_stitch_img.shape[1]))
			
			

	def align_btn_23_clicked(self, parent=None):
		print "Align Btn 23 Press"
		self.alignFlag23 = True

		print len(self.h_matrix_list)
		self.recaptureFrames()
		prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
		prev_right_img = stitch.get_stitched_right_image(self.images[2],prev_right_img,self.h_matrix_list[1])


		img1 = prev_right_img
		img2 = self.images[3]

		min_error_width = 500
		min_error_height = 100
		video_height = 717

		# Get width and height of input images	
		w1,h1 = img1.shape[:2]
		w2,h2 = img2.shape[:2]

		# Get the canvas dimesions
		img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

		optimal_transform_matrix_change_value = 100
		optimal_transform_matrix = None
		optimal_x_max = 0
		optimal_x_min = 0
		optimal_y_max = 0
		optimal_y_min = 0	


		while(self.alignFlag23):
			QtGui.QApplication.processEvents()
			self.recaptureFrames()
			prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
			prev_right_img = stitch.get_stitched_right_image(self.images[2],prev_right_img,self.h_matrix_list[1])



			img1 = prev_right_img
			img2 = self.images[3]


			feature_str = ""

			MIN_MATCH_COUNT = 10

			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create() 

			# find the keypoints and descriptors with SIFT
			k1, d1 = sift.detectAndCompute(img1,None)
			k2, d2 = sift.detectAndCompute(img2,None)




			# Bruteforce matcher on the descriptors
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(d1,d2, k=2)

			# Make sure that the matches are good
			verify_ratio = 0.8 # Source: stackoverflow
			verified_matches = []
			for m1,m2 in matches:
				# Add to array only if it's a good match
				if m1.distance < 0.8 * m2.distance:
					verified_matches.append(m1)

			# Mimnum number of matches

			if len(verified_matches) > MIN_MATCH_COUNT:
	
				# Array to store matching points
				img1_pts = []
				img2_pts = []

				# Add matching points to array
				for match in verified_matches:
					img1_pts.append(k1[match.queryIdx].pt)
					img2_pts.append(k2[match.trainIdx].pt)
				img1_pts = np.float32(img1_pts).reshape(-1,1,2)
				img2_pts = np.float32(img2_pts).reshape(-1,1,2)
	
				# Compute homography matrix
				M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
				feature_str = "Good :"+str(len(verified_matches))
			else:
				#print 'Error: Not enough matches'
				feature_str = "Not Good :"+str(len(verified_matches))+"/"+str(MIN_MATCH_COUNT)


			#print feature_str



			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = mask.ravel().tolist(), # draw only inliers
					   flags = 2)

			feature_match_img = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)



			# Get relative perspective of second image
			img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

			# Resulting dimensions
			result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)



			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

			# Create output array after affine transformation 
			transform_dist = [-x_min,-y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

			original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])
	
			transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))

			


			# Resulting dimensions
			result_dims = np.concatenate( (img2_dims, img1_dims), axis = 0)

			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
			

			temp_stitch_img = stitch.get_stitched_right_image(img2,img1,M)

			if optimal_transform_matrix == None:
				print optimal_transform_matrix
				optimal_transform_matrix = M
				optimal_x_max = x_max
				optimal_x_min = x_min
				optimal_y_max = y_max
				optimal_y_min = y_min

			print str(temp_stitch_img.shape[1]) + " " + str(self.stitch_width_list[2])+ " " + str(temp_stitch_img.shape[0]) + " " + str(video_height)


			if abs(temp_stitch_img.shape[1] - self.stitch_width_list[2]) < 100 and min_error_width > abs(temp_stitch_img.shape[1] - self.stitch_width_list[2]):
				#if abs(temp_stitch_img.shape[0] - video_height) < 10 and min_error_height > abs(temp_stitch_img.shape[0] - video_height): 
					print "12 New Min Value : "+str(transform_matrix_change_value)
					optimal_transform_matrix_change_value = transform_matrix_change_value

					min_error_width = abs(temp_stitch_img.shape[1] - self.stitch_width_list[0])
					min_error_height = abs(temp_stitch_img.shape[0] - video_height)

					optimal_transform_matrix = M
					optimal_x_max = x_max
					optimal_x_min = x_min
					optimal_y_max = y_max
					optimal_y_min = y_min

					if len(self.h_matrix_list) == 2:
						print "23 H Matrix List Append : 2"
						self.h_matrix_list.append(M)
						self.min_max_list.append([x_min, x_max, y_min, y_max])
					else:
						print "23 H Matrix List Value Changed : 2"
						self.h_matrix_list[2] = M
						self.min_max_list[2] = [x_min, x_max, y_min, y_max]



			# Create output array after affine transformation 
			transform_dist = [-optimal_x_min,-optimal_y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

			# Warp images to get the resulting image
			#stitch_img = cv2.warpPerspective(img1, transform_array.dot(optimal_transform_matrix),(optimal_x_max-optimal_x_min, optimal_y_max-optimal_y_min))
			#stitch_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img2
			stitch_img = stitch.get_stitched_image(img2,img1,optimal_transform_matrix)


			


			cv2.imshow("Feature_0_1",imutils.resize(feature_match_img,width=700))
			cv2.imshow("Stitch_0_1",imutils.resize(stitch_img,width=700))
			cv2.waitKey(1)

			self.ui.align_label_23.setText(feature_str+"\n"+str(transform_matrix_change_value)+"\n"+str(temp_stitch_img.shape[1]))



	def align_btn_34_clicked(self, parent=None):
		print "Align Btn 34 Press"
		self.alignFlag34 = True
		
		print len(self.h_matrix_list)
		self.recaptureFrames()
		prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
		prev_right_img = stitch.get_stitched_right_image(self.images[2],self.images[1],self.h_matrix_list[1])
		prev_right_img = stitch.get_stitched_right_image(self.images[3],prev_right_img,self.h_matrix_list[2])


		img1 = prev_right_img
		img2 = self.images[3]

		min_error_width = 500
		min_error_height = 100
		video_height = 717

		# Get width and height of input images	
		w1,h1 = img1.shape[:2]
		w2,h2 = img2.shape[:2]

		# Get the canvas dimesions
		img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

		optimal_transform_matrix_change_value = 100
		optimal_transform_matrix = None
		optimal_x_max = 0
		optimal_x_min = 0
		optimal_y_max = 0
		optimal_y_min = 0	


		while(self.alignFlag34):
			QtGui.QApplication.processEvents()
			self.recaptureFrames()
			prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
			prev_right_img = stitch.get_stitched_right_image(self.images[2],self.images[1],self.h_matrix_list[1])
			prev_right_img = stitch.get_stitched_right_image(self.images[3],prev_right_img,self.h_matrix_list[2])



			img1 = prev_right_img
			img2 = self.images[4]


			feature_str = ""

			MIN_MATCH_COUNT = 10

			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create() 

			# find the keypoints and descriptors with SIFT
			k1, d1 = sift.detectAndCompute(img1,None)
			k2, d2 = sift.detectAndCompute(img2,None)




			# Bruteforce matcher on the descriptors
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(d1,d2, k=2)

			# Make sure that the matches are good
			verify_ratio = 0.8 # Source: stackoverflow
			verified_matches = []
			for m1,m2 in matches:
				# Add to array only if it's a good match
				if m1.distance < 0.8 * m2.distance:
					verified_matches.append(m1)

			# Mimnum number of matches

			if len(verified_matches) > MIN_MATCH_COUNT:

				# Array to store matching points
				img1_pts = []
				img2_pts = []

				# Add matching points to array
				for match in verified_matches:
					img1_pts.append(k1[match.queryIdx].pt)
					img2_pts.append(k2[match.trainIdx].pt)
				img1_pts = np.float32(img1_pts).reshape(-1,1,2)
				img2_pts = np.float32(img2_pts).reshape(-1,1,2)

				# Compute homography matrix
				M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
				feature_str = "Good :"+str(len(verified_matches))
			else:
				#print 'Error: Not enough matches'
				feature_str = "Not Good :"+str(len(verified_matches))+"/"+str(MIN_MATCH_COUNT)


			#print feature_str



			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = mask.ravel().tolist(), # draw only inliers
					   flags = 2)

			feature_match_img = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)



			# Get relative perspective of second image
			img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

			# Resulting dimensions
			result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)



			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

			# Create output array after affine transformation 
			transform_dist = [-x_min,-y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

			original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])

			transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))




			# Resulting dimensions
			result_dims = np.concatenate( (img2_dims, img1_dims), axis = 0)

			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)


			temp_stitch_img = stitch.get_stitched_right_image(img2,img1,M)

			if optimal_transform_matrix == None:
				print optimal_transform_matrix
				optimal_transform_matrix = M
				optimal_x_max = x_max
				optimal_x_min = x_min
				optimal_y_max = y_max
				optimal_y_min = y_min

			print str(temp_stitch_img.shape[1]) + " " + str(self.stitch_width_list[3])+ " " + str(temp_stitch_img.shape[0]) + " " + str(video_height)

			if abs(temp_stitch_img.shape[1] - self.stitch_width_list[3]) < 100 and min_error_width > abs(temp_stitch_img.shape[1] - self.stitch_width_list[3]):
				#if abs(temp_stitch_img.shape[0] - video_height) < 10 and min_error_height > abs(temp_stitch_img.shape[0] - video_height): 
					print "34 New Min Value : "+str(transform_matrix_change_value)
					optimal_transform_matrix_change_value = transform_matrix_change_value

					min_error_width = abs(temp_stitch_img.shape[1] - self.stitch_width_list[0])
					min_error_height = abs(temp_stitch_img.shape[0] - video_height)

					optimal_transform_matrix = M
					optimal_x_max = x_max
					optimal_x_min = x_min
					optimal_y_max = y_max
					optimal_y_min = y_min

					if len(self.h_matrix_list) == 3:
						print "34 H Matrix List Append : 3"
						self.h_matrix_list.append(M)
						self.min_max_list.append([x_min, x_max, y_min, y_max])
					else:
						print "34 H Matrix List Value Changed : 3"
						self.h_matrix_list[3] = M
						self.min_max_list[3] = [x_min, x_max, y_min, y_max]



			# Create output array after affine transformation 
			transform_dist = [-optimal_x_min,-optimal_y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

			# Warp images to get the resulting image
			#stitch_img = cv2.warpPerspective(img1, transform_array.dot(optimal_transform_matrix),(optimal_x_max-optimal_x_min, optimal_y_max-optimal_y_min))
			#stitch_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img2
			stitch_img = stitch.get_stitched_image(img2,img1,optimal_transform_matrix)





			cv2.imshow("Feature_0_1",imutils.resize(feature_match_img,width=700))
			cv2.imshow("Stitch_0_1",imutils.resize(stitch_img,width=700))
			cv2.waitKey(1)

			self.ui.align_label_34.setText(feature_str+"\n"+str(transform_matrix_change_value)+"\n"+str(temp_stitch_img.shape[1]))



	def align_btn_45_clicked(self, parent=None):
		print "Align Btn 45 Press"
		self.alignFlag45 = True
		print len(self.h_matrix_list)
		self.recaptureFrames()
		prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
		prev_right_img = stitch.get_stitched_right_image(self.images[2],self.images[1],self.h_matrix_list[1])
		prev_right_img = stitch.get_stitched_right_image(self.images[3],self.images[2],self.h_matrix_list[2])
		prev_right_img = stitch.get_stitched_right_image(self.images[4],prev_right_img,self.h_matrix_list[3])


		img1 = prev_right_img
		img2 = self.images[5]

		min_error_width = 500
		min_error_height = 100
		video_height = 717

		# Get width and height of input images	
		w1,h1 = img1.shape[:2]
		w2,h2 = img2.shape[:2]

		# Get the canvas dimesions
		img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

		optimal_transform_matrix_change_value = 100
		optimal_transform_matrix = None
		optimal_x_max = 0
		optimal_x_min = 0
		optimal_y_max = 0
		optimal_y_min = 0	


		while(self.alignFlag45):
			QtGui.QApplication.processEvents()
			self.recaptureFrames()
			prev_right_img = stitch.get_stitched_right_image(self.images[1],self.images[0],self.h_matrix_list[0])
			prev_right_img = stitch.get_stitched_right_image(self.images[2],self.images[1],self.h_matrix_list[1])
			prev_right_img = stitch.get_stitched_right_image(self.images[3],self.images[2],self.h_matrix_list[2])
			prev_right_img = stitch.get_stitched_right_image(self.images[4],prev_right_img,self.h_matrix_list[3])



			img1 = prev_right_img
			img2 = self.images[5]


			feature_str = ""

			MIN_MATCH_COUNT = 10

			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create() 

			# find the keypoints and descriptors with SIFT
			k1, d1 = sift.detectAndCompute(img1,None)
			k2, d2 = sift.detectAndCompute(img2,None)




			# Bruteforce matcher on the descriptors
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(d1,d2, k=2)

			# Make sure that the matches are good
			verify_ratio = 0.8 # Source: stackoverflow
			verified_matches = []
			for m1,m2 in matches:
				# Add to array only if it's a good match
				if m1.distance < 0.8 * m2.distance:
					verified_matches.append(m1)

			# Mimnum number of matches

			if len(verified_matches) > MIN_MATCH_COUNT:

				# Array to store matching points
				img1_pts = []
				img2_pts = []

				# Add matching points to array
				for match in verified_matches:
					img1_pts.append(k1[match.queryIdx].pt)
					img2_pts.append(k2[match.trainIdx].pt)
				img1_pts = np.float32(img1_pts).reshape(-1,1,2)
				img2_pts = np.float32(img2_pts).reshape(-1,1,2)

				# Compute homography matrix
				M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
				feature_str = "Good :"+str(len(verified_matches))
			else:
				#print 'Error: Not enough matches'
				feature_str = "Not Good :"+str(len(verified_matches))+"/"+str(MIN_MATCH_COUNT)


			#print feature_str



			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = mask.ravel().tolist(), # draw only inliers
					   flags = 2)

			feature_match_img = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)



			# Get relative perspective of second image
			img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

			# Resulting dimensions
			result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)



			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

			# Create output array after affine transformation 
			transform_dist = [-x_min,-y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

			original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])

			transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))




			# Resulting dimensions
			result_dims = np.concatenate( (img2_dims, img1_dims), axis = 0)

			# Getting images together
			# Calculate dimensions of match points
			[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)


			temp_stitch_img = stitch.get_stitched_right_image(img2,img1,M)

			if optimal_transform_matrix == None:
				print optimal_transform_matrix
				optimal_transform_matrix = M
				optimal_x_max = x_max
				optimal_x_min = x_min
				optimal_y_max = y_max
				optimal_y_min = y_min

			print str(temp_stitch_img.shape[1]) + " " + str(self.stitch_width_list[4])+ " " + str(temp_stitch_img.shape[0]) + " " + str(video_height)

			if abs(temp_stitch_img.shape[1] - self.stitch_width_list[4]) < 100 and min_error_width > abs(temp_stitch_img.shape[1] - self.stitch_width_list[4]):
				#if abs(temp_stitch_img.shape[0] - video_height) < 10 and min_error_height > abs(temp_stitch_img.shape[0] - video_height): 
					print "45 New Min Value : "+str(transform_matrix_change_value)
					optimal_transform_matrix_change_value = transform_matrix_change_value

					min_error_width = abs(temp_stitch_img.shape[1] - self.stitch_width_list[4])
					min_error_height = abs(temp_stitch_img.shape[0] - video_height)

					optimal_transform_matrix = M
					optimal_x_max = x_max
					optimal_x_min = x_min
					optimal_y_max = y_max
					optimal_y_min = y_min

					if len(self.h_matrix_list) == 4:
						print "45 H Matrix List Append : 4"
						self.h_matrix_list.append(M)
						self.min_max_list.append([x_min, x_max, y_min, y_max])
					else:
						print "45 H Matrix List Value Changed : 4"
						self.h_matrix_list[4] = M
						self.min_max_list[4] = [x_min, x_max, y_min, y_max]



			# Create output array after affine transformation 
			transform_dist = [-optimal_x_min,-optimal_y_min]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 

			# Warp images to get the resulting image
			#stitch_img = cv2.warpPerspective(img1, transform_array.dot(optimal_transform_matrix),(optimal_x_max-optimal_x_min, optimal_y_max-optimal_y_min))
			#stitch_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img2
			stitch_img = stitch.get_stitched_image(img2,img1,optimal_transform_matrix)





			cv2.imshow("Feature_0_1",imutils.resize(feature_match_img,width=700))
			cv2.imshow("Stitch_0_1",imutils.resize(stitch_img,width=700))
			cv2.waitKey(1)

			self.ui.align_label_45.setText(feature_str+"\n"+str(transform_matrix_change_value)+"\n"+str(temp_stitch_img.shape[1]))




	def align_stop_btn_01_clicked(self, parent=None):
		print "Align Stop Btn 01 Press"
		self.alignFlag01 = False

	def align_stop_btn_12_clicked(self, parent=None):
		print "Align Stop Btn 12 Press"
		self.alignFlag12 = False

	def align_stop_btn_23_clicked(self, parent=None):
		print "Align Stop Btn 23 Press"
		self.alignFlag23 = False

	def align_stop_btn_34_clicked(self, parent=None):
		print "Align Stop Btn 34 Press"
		self.alignFlag34 = False

	def align_stop_btn_45_clicked(self, parent=None):
		print "Align Stop Btn 45 Press"
		self.alignFlag45 = False




	def img_stitch_btn_clicked(self, parent=None):
		print "Image Stitch Btn Press"
		self.ui.img_stitch_label.setText("Start")
		self.recaptureFrames()

		stitch_img_right = self.images[0]
		stitch_img = self.images[0]

		for index in range(len(self.images)):
			QtGui.QApplication.processEvents()
			if index != 0:
				M =  stitch.get_sift_homography(stitch_img,self.images[index])
				x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(self.images[index],stitch_img,M)


				temp_stitch_img = stitch.get_stitched_right_image(self.images[index],stitch_img_right,M)
				print index, x_min, x_max, y_min, y_max, temp_stitch_img.shape[1]
				self.ui.img_stitch_label.setText(str(index)+" "+str(x_min)+" "+str(x_max)+" "+str(y_min)+" "+str(y_max)+" "+str(temp_stitch_img.shape[1]))
			
				#while (abs(temp_stitch_img.shape[1] - 1050*(index+1)) > 150) :
				while (abs(temp_stitch_img.shape[1] - self.stitch_width_list[index-1]) > 100) :
					QtGui.QApplication.processEvents()
					self.recaptureFrames()


					stitch_img_right = self.images[0]
					stitch_img = self.images[0]

					for sub_index in range(index):
						QtGui.QApplication.processEvents()
						print str(index)+"  -> Refresh Image : "+str(sub_index)
						if sub_index != 0:
							
							temp_stitch_img = stitch.get_stitched_right_image(self.images[sub_index],stitch_img_right,self.h_matrix_list[sub_index-1])
							cv2.imwrite("temp_stitch_%d_right.jpg"%(sub_index),stitch_img_right)


					M =  stitch.get_sift_homography(temp_stitch_img,self.images[index])
					x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(self.images[index],temp_stitch_img,M)
					temp_stitch_img = stitch.get_stitched_right_image(self.images[index],temp_stitch_img,M)
					print index, x_min, x_max, y_min, y_max, temp_stitch_img.shape[1], self.stitch_width_list[index-1]

				self.h_matrix_list.append(M)
				self.min_max_list.append([x_min, x_max, y_min, y_max])

				stitch_img = stitch.get_stitched_image(self.images[index],stitch_img,M)
				stitch_img_right = stitch.get_stitched_right_image(self.images[index],stitch_img_right,M)
				cv2.imwrite("stitch_%d_right.jpg"%(index),stitch_img_right)
				cv2.imwrite("stitch_%d.jpg"%(index),stitch_img)
				#cv2.imshow("Stitched_Right_%d"%(index), imutils.resize(stitch_img_right,width=1200))
				#cv2.imshow("Stitched_%d"%(index), imutils.resize(stitch_img,width=1200))
				#cv2.waitKey(1)
		self.ui.img_stitch_label.setText("Finish Stitch")
		
		for i in range(len(self.images)-1):	
			(xmin,xmax,ymin,ymax) = self.min_max_list[i]
			transform_dist = [-xmin,-ymin]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]]) 
			self.transform_matrix_list.append(transform_array.dot(self.h_matrix_list[i]))
			#print transform_array.dot(h_matrix_list[i])

		for j in range(len(self.images)-1):
			self.final_transform_matrix_list.append(transform_matrix_list[-1])

		for i in range(len(self.images)-1):
			for j in range((len(self.images)-3-i),-1,-1):
				self.final_transform_matrix_list[i] = np.dot(self.final_transform_matrix_list[i],self.transform_matrix_list[j])


		



		# Get Total Width & Total Height From Last Min Max Values
		self.total_width = self.min_max_list[-1][1] - self.min_max_list[-1][0]
		self.total_height = self.min_max_list[-1][3] - self.min_max_list[-1][2]

		f = open('basic_params.pckl', 'w')
		pickle.dump((self.h_matrix_list, self.min_max_list, self.transform_matrix_list, self.final_transform_matrix_list, self.total_width, self.total_height), f)
		f.close()

		self.ui.img_stitch_label.setText("Save Params")
	



	def save_params_btn_clicked(self, parent=None):
		print "Save Params Btn Press"
		f = open('basic_params.pckl', 'w')
		pickle.dump((self.h_matrix_list, self.min_max_list), f)
		f.close()
		

		

	def load_params_btn_clicked(self, parent=None):
		print "Load Params Btn Press"
		f = open('basic_params.pckl')
		#self.h_matrix_list, self.min_max_list, self.transform_matrix_list, self.final_transform_matrix_list, self.total_width, self.total_height = pickle.load(f)
		self.h_matrix_list, self.min_max_list = pickle.load(f)
		f.close()




	def create_mask_btn_clicked(self, parent=None):
		print "Create Mask Btn Press"
		
		self.recaptureFrames()
		stitch_img = self.images[0]
		self.min_max_list = list()
		for index in range(len(self.images)):
			if index != 0:
				M =  self.h_matrix_list[index-1]
				x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(self.images[index],stitch_img,M)

				print index, x_min, x_max, y_min, y_max
				self.min_max_list.append([x_min, x_max, y_min, y_max])
				stitch_img = stitch.get_stitched_image(self.images[index],stitch_img,M)
				cv2.imwrite("stitch_%d.jpg"%(index),stitch_img)
		

		self.transform_matrix_list = list()


		for i in range(len(self.images)-1):	
			(xmin,xmax,ymin,ymax) = self.min_max_list[i]
			transform_dist = [-xmin,-ymin]
			transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])
			self.transform_matrix_list.append(transform_array.dot(self.h_matrix_list[i]))
			#print transform_array.dot(h_matrix_list[i])



		


		self.final_transform_matrix_list = list()


		for i in range(len(self.images)-1):
			self.final_transform_matrix_list.append(self.transform_matrix_list[i])
			print i,range(i+1,len(self.transform_matrix_list),1)
			for j in range(i+1,len(self.transform_matrix_list),1):
				print i,j
				self.final_transform_matrix_list[i] = np.dot(self.final_transform_matrix_list[i],self.transform_matrix_list[j])




		# Get Total Width & Total Height From Last Min Max Values
		self.total_width = self.min_max_list[-1][1] - self.min_max_list[-1][0]
		self.total_height = self.min_max_list[-1][3] - self.min_max_list[-1][2]

		print self.total_width





		# Create Transform Images
		self.trans_images = list()
		for i in range(len(self.images)):
			if i == 0:
				self.trans_images.append(cv2.warpPerspective(self.images[i], self.final_transform_matrix_list[i],(self.total_width,self.total_height)))

			elif i < (len(self.images)-1):
				self.trans_images.append(cv2.warpPerspective(cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height)), self.final_transform_matrix_list[i],(self.total_width,self.total_height)))

			else:
				self.trans_images.append(cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height)))


		cv2.imwrite('trans0.jpg',self.trans_images[0])
		cv2.imwrite('trans1.jpg',self.trans_images[1])
		cv2.imwrite('trans2.jpg',self.trans_images[2])
		cv2.imwrite('trans3.jpg',self.trans_images[3])
		cv2.imwrite('trans4.jpg',self.trans_images[4])
		cv2.imwrite('trans5.jpg',self.trans_images[5])



		start_end_points_of_each_trans_image_list = list()

		for i in range(len(self.images)):

			start = 0
			end = 0

			for x in range(self.trans_images[i].shape[1]):

				if start == 0 and np.sum(self.trans_images[i][:,x]) != 0:
					start = x
				elif start != 0 and np.sum(self.trans_images[i][:,x]) == 0:
					end = x
					break

			start_end_points_of_each_trans_image_list.append([start,end])
	
		start_end_points_of_each_trans_image_list[-1][1] = self.total_width



		self.cut_position_list = list()

		i=0
		# Find Overlap Positions

		try:
			for i in range(len(self.images)-1):
				compare_value_list = list()
	
				fromX = start_end_points_of_each_trans_image_list[i+1][0]+30
				toX = min(start_end_points_of_each_trans_image_list[i+1][0]+500,start_end_points_of_each_trans_image_list[i][1])
				#print i, fromX, toX
				#print fromX, toX
				for x in range(fromX,toX):
					crop_imageA = self.trans_images[i][:,x]
					crop_imageB = self.trans_images[i+1][:,x]

					err = np.sum((crop_imageA.astype('float')-crop_imageB.astype('float'))**2)
					err /= float(crop_imageA.shape[0] * crop_imageA.shape[1])    
				    
					compare_value_list.append(err)   
				#print compare_value_list

				min_err_px = compare_value_list.index(min(compare_value_list))
				self.cut_position_list.append(min_err_px+fromX)
				print (min_err_px+fromX)
		except:
			print "Error Occurred"


		after_mask_trans_image_array2 = np.zeros((self.total_height,self.total_width,3))

		after_mask_trans_image_array2[:,0:self.cut_position_list[0],:] = self.trans_images[0][:,0:self.cut_position_list[0],:]
		after_mask_trans_image_array2[:,self.cut_position_list[0]:self.cut_position_list[1],:] = self.trans_images[1][:,self.cut_position_list[0]:self.cut_position_list[1],:]
		after_mask_trans_image_array2[:,self.cut_position_list[1]:self.cut_position_list[2],:] = self.trans_images[2][:,self.cut_position_list[1]:self.cut_position_list[2],:]
		after_mask_trans_image_array2[:,self.cut_position_list[2]:self.cut_position_list[3],:] = self.trans_images[3][:,self.cut_position_list[2]:self.cut_position_list[3],:]
		after_mask_trans_image_array2[:,self.cut_position_list[3]:self.cut_position_list[4],:] = self.trans_images[4][:,self.cut_position_list[3]:self.cut_position_list[4],:]
		after_mask_trans_image_array2[:,self.cut_position_list[4]:,:] = self.trans_images[5][:,self.cut_position_list[4]:,:]

		cv2.imwrite('final_stitched_without_seam.jpg',after_mask_trans_image_array2)

		



		self.seam_list = list()

		for i in range(len(self.images)-1):
			fromX = self.cut_position_list[i]-22
			toX = fromX + 25

		    	imgSeam = self.trans_images[i]

			img_gray = cv2.cvtColor(imgSeam[:,fromX:toX,:], cv2.COLOR_RGB2GRAY)
			img_en = seamfunc.getEnergyImage(img_gray)

			self.seam_list.append(seamfunc.extractSeam(img_en)+fromX)
	








		


		basic_mask_list = list()


		
		# Create Basic Mask and Reverse of Basic Mask
		for i in range(len(self.trans_images)):
		    if i < len(self.trans_images)-1:
			basic_mask_list.append(self.createBasicMask(i))
		    else:
			mask_ones = np.dstack((np.ones(self.trans_images[i].shape[:2]).astype(int),np.ones(self.trans_images[i].shape[:2]).astype(int),np.ones(self.trans_images[i].shape[:2]).astype(int)))
			basic_mask_list.append((mask_ones,1-mask_ones))



		

		# Create Combined Mask for Each Image
		self.combined_mask_list = list()
		for i in range(len(self.trans_images)):
			if i == 0:
				self.combined_mask_list.append(basic_mask_list[i][0])
			elif i < len(self.trans_images)-1:
				self.combined_mask_list.append(basic_mask_list[i-1][1]*basic_mask_list[i][0])
			else:
				self.combined_mask_list.append(basic_mask_list[i-1][1])

			cv2.imwrite("mask_%d.jpg"%(i),self.combined_mask_list[i]*255)


		


		after_mask_trans_image_array = np.zeros((len(self.trans_images),self.total_height,self.total_width,3))


		for i in range(len(self.trans_images)):
		    after_mask_trans_image_array[i,:,:,:] = self.applyMaskToImage(i)


		final_combined_image = np.sum(after_mask_trans_image_array, 0)


		cv2.imwrite('final_stitched_with_seam.jpg',final_combined_image)




	def img_stitch_without_seam_btn_clicked(self, parent=None):
		print "Image Stitch Without Seam Btn Press"
		self.recaptureFrames()
		stitch_img = self.images[0]
		for index in range(len(self.images)):
			if index != 0:
				M =  self.h_matrix_list[index-1]
				x_min, x_max, y_min, y_max = self.min_max_list[index-1]

				print index, x_min, x_max, y_min, y_max
				
				stitch_img = stitch.get_stitched_image(self.images[index],stitch_img,M)
				cv2.imwrite("stitch_%d.jpg"%(index),stitch_img)
		


	def img_stitch_with_seam_btn_clicked(self, parent=None):
		print "Image Stitch With Seam Btn Press"
		print len(self.h_matrix_list)



	def video_stitch_without_seam_btn_clicked(self, parent=None):
		print "Video Stitch Without Seam Btn Press"
		self.recaptureFrames()

		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out0 = cv2.VideoWriter('video_stitch_without_seam.avi', fourcc, 5.0, (self.total_width,self.total_height))

		self.videoStitchFlag = True


		after_mask_trans_image_array2 = np.zeros((self.total_height,self.total_width,3))

		while self.videoStitchFlag == True:
			start = timer()

			self.recaptureFrames()
			self.trans_images = list()
			
			for i in range(len(self.images)):
				if i == 0:
					self.trans_images.append(cv2.warpPerspective(self.images[i], self.final_transform_matrix_list[i],(self.total_width,self.total_height)))

				elif i < (len(self.images)-1):
					self.trans_images.append(cv2.warpPerspective(cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height)), self.final_transform_matrix_list[i],(self.total_width,self.total_height)))

				else:
					self.trans_images.append(cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height)))



			after_mask_trans_image_array2 = np.zeros((self.total_height,self.total_width,3))

			after_mask_trans_image_array2[:,0:self.cut_position_list[0],:] = self.trans_images[0][:,0:self.cut_position_list[0],:]
			after_mask_trans_image_array2[:,self.cut_position_list[0]:self.cut_position_list[1],:] = self.trans_images[1][:,self.cut_position_list[0]:self.cut_position_list[1],:]
			after_mask_trans_image_array2[:,self.cut_position_list[1]:self.cut_position_list[2],:] = self.trans_images[2][:,self.cut_position_list[1]:self.cut_position_list[2],:]
			after_mask_trans_image_array2[:,self.cut_position_list[2]:self.cut_position_list[3],:] = self.trans_images[3][:,self.cut_position_list[2]:self.cut_position_list[3],:]
			after_mask_trans_image_array2[:,self.cut_position_list[3]:self.cut_position_list[4],:] = self.trans_images[4][:,self.cut_position_list[3]:self.cut_position_list[4],:]
			after_mask_trans_image_array2[:,self.cut_position_list[4]:,:] = self.trans_images[5][:,self.cut_position_list[4]:,:]

			end = timer()
			fps_str = "%.1f FPS"%(1 / (end-start))
			img_with_fps = cv2.putText(after_mask_trans_image_array2.astype(self.images[0].dtype), fps_str, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),4)
			#cv2.imshow('frame',imutils.resize(after_mask_trans_image_array2.astype(self.images[0].dtype), height=600))
			cv2.imshow('frame',imutils.resize(img_with_fps, width=1800))
			#out0.write(after_mask_trans_image_array2.astype(self.images[0].dtype))
			out0.write(img_with_fps.astype(self.images[0].dtype))

			cv2.waitKey(1)



	def video_stitch_with_seam_btn_clicked(self, parent=None):
		print "Video Stitch With Seam Btn Press"
		#self.videoStitchFlag = False

		
		self.recaptureFrames()

		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out0 = cv2.VideoWriter('video_stitch_with_seam.avi', fourcc, 2.0, (self.total_width,self.total_height))

		self.videoStitchFlag = True


		after_mask_trans_image_array2 = np.zeros((self.total_height,self.total_width,3))

		while self.videoStitchFlag == True:
			start = timer()
			self.recaptureFrames()
			self.trans_images = list()



			# Image Warping with Transform Matrix
			for i in range(len(self.images)):
	
				if i == 0:		
					after_mask_trans_image_array2 = cv2.warpPerspective(self.images[i], self.final_transform_matrix_list[i],(self.total_width,self.total_height))*self.combined_mask_list[i]

				elif i < (len(self.images)-1):
					after_mask_trans_image_array2 = after_mask_trans_image_array2 + cv2.warpPerspective(cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height)), self.final_transform_matrix_list[i],(self.total_width,self.total_height))*self.combined_mask_list[i]

				else:
					after_mask_trans_image_array2 = after_mask_trans_image_array2 + cv2.warpAffine(self.images[i],np.float32([[1,0,-self.min_max_list[i-1][0]],[0,1,-self.min_max_list[i-1][2]]]),(self.total_width,self.total_height))*self.combined_mask_list[i]
			

			end = timer()
			fps_str = "%.1f FPS"%(1 / (end-start))
			img_with_fps = cv2.putText(after_mask_trans_image_array2.astype(self.images[0].dtype), fps_str, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),4)
			#cv2.imshow('frame',imutils.resize(after_mask_trans_image_array2.astype(self.images[0].dtype), height=600))
			cv2.imshow('frame',imutils.resize(img_with_fps, width=1800))
			out0.write(img_with_fps.astype(self.images[0].dtype))
			cv2.waitKey(1)






	# Preparing Cylindrical Projection
	def cylinMapping(self):

		frame_0 = self.cap_0.read()

		def makeCylinMap(y, x):
		    #print x, y    
		    
		    ximg = toXimg(y,x)
		    yimg = toYimg(y,x)
		    return np.dstack((ximg, yimg)).astype(np.int16)


		def makeOriMap(y,x):
		    return np.dstack((x,y)).astype(np.int16)
		    

			    
		out_img = np.multiply(frame_0,0)

		xdim = frame_0.shape[1]
		ydim = frame_0.shape[0]

		xc = xdim / 2
		yc = ydim / 2


		#f = np.float(1150)
		f = np.float(1100)
		k1 = 0.03
		k2 = 0.03



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

		#map_ori_xy = np.fromfunction(makeOriMap, img.shape[:2], dtype=np.int16)
		map_cylin_xy = np.fromfunction(makeCylinMap, frame_0.shape[:2], dtype=np.int16)


		img_mapped_0 = cv2.remap(frame_0, (map_cylin_xy), None, False)

		## Find Contour Region
		gray = cv2.cvtColor(img_mapped_0,cv2.COLOR_BGR2GRAY)
		_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

		_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		cylinX,cylinY,cylinW,cylinH = cv2.boundingRect(cnt)

		return map_cylin_xy, cylinX, cylinY, cylinW, cylinH


	def getStitchInfo(self, imgLeft, imgRight):
		img1 = imgLeft
		img2 = imgRight

		M =  stitch.get_sift_homography(img1,img2)
		x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(img2,img1,M)

		# Create output array after affine transformation 
		transform_dist = [-x_min,-y_min]
		transform_array = np.array([[1, 0, transform_dist[0]],[0, 1, transform_dist[1]],[0,0,1]])

		original_trans = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0,0.0,1.0]])
		
		transform_matrix_change_value = sum(sum(abs(transform_array.dot(M) - original_trans)))
		return transform_matrix_change_value


	def findMaskLeftEnd(self, mask):
		    for x in range(mask.shape[1]):
			if np.prod(mask[:,x]) != 0:
			    return x
			    break

	def createBasicMask(self, mask_num):
	    i = mask_num
	    mask_layer = np.fromfunction(lambda y,x : self.seam_list[i][y] >= x, self.trans_images[i].shape[:2], dtype=int)
	    mask_layer = mask_layer.astype(int)
	    mask = np.dstack((mask_layer,mask_layer,mask_layer))
	    
	    return mask,1-mask

	def applyMaskToImage(self, num):
	    return self.trans_images[num][:,:,:]*self.combined_mask_list[num]





	def recaptureFrames(self) :
		QtGui.QApplication.processEvents()
		frame_0 = self.cap_0.read()
		frame_1 = self.cap_1.read()    
		frame_2 = self.cap_2.read()
		frame_3 = self.cap_3.read()
		frame_4 = self.cap_4.read()
		frame_5 = self.cap_5.read()
		

		img_mapped_0 = cv2.remap(frame_0, (self.map_cylin_xy), None, False)
		img_mapped_1 = cv2.remap(frame_1, (self.map_cylin_xy), None, False)
		img_mapped_2 = cv2.remap(frame_2, (self.map_cylin_xy), None, False)
		img_mapped_3 = cv2.remap(frame_3, (self.map_cylin_xy), None, False)
		img_mapped_4 = cv2.remap(frame_4, (self.map_cylin_xy), None, False)
		img_mapped_5 = cv2.remap(frame_5, (self.map_cylin_xy), None, False)


		crop_0 = img_mapped_0[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_1 = img_mapped_1[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_2 = img_mapped_2[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_3 = img_mapped_3[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_4 = img_mapped_4[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]
		crop_5 = img_mapped_5[self.cylinY:self.cylinY+self.cylinH,self.cylinX+10:self.cylinX+self.cylinW-10]

		

		if len(self.images) == 0:
			self.images.append(crop_0)
			self.images.append(crop_1)
			self.images.append(crop_2)
			self.images.append(crop_3)
			self.images.append(crop_4)
			self.images.append(crop_5)
		else :					
			self.images[0] = crop_0
			self.images[1] = crop_1
			self.images[2] = crop_2
			self.images[3] = crop_3
			self.images[4] = crop_4
			self.images[5] = crop_5



if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = MyGui()
	myapp.show()
	sys.exit(app.exec_())
