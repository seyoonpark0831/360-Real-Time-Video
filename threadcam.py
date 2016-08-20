from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
from timeit import default_timer as timer


ws_0 = WebcamVideoStream(src=0)
ws_0.stream.set(3,1280)
ws_0.stream.set(4,720)
ws_0.stream.set(5,30)
vs_0 = ws_0.start()

ws_1 = WebcamVideoStream(src=1)
ws_1.stream.set(3,1280)
ws_1.stream.set(4,720)
ws_1.stream.set(5,30)
vs_1 = ws_1.start()


while True:
	start = timer()
	frame_0 = vs_0.read()
	frame_1 = vs_1.read()
	#frame_2 = vs_2.read()
	#frame_3 = vs_3.read()
	#frame_4 = vs_4.read()
	#frame_5 = vs_5.read()
	cv2.imshow("F", frame_1)
	cv2.imshow("F0", frame_0)
	#cv2.imshow("F", np.concatenate((frame_0,frame_1),axis=1))
	end = timer()
	print (1/(end-start))
	#cv2.imshow("Frame0",imutils.resize(np.concatenate((np.concatenate((frame_0, frame_1, frame_2), axis=1),np.concatenate((frame_2, frame_3, frame_4), axis=1),np.concatenate((frame_3, frame_4, frame_5), axis=1)), axis=0),width=1000))
	if cv2.waitKey(1) & 0xFF == ord('z'):
		break

vs_0.stop()
vs_1.stop()
vs_2.stop()
vs_3.stop()
vs_4.stop()
vs_5.stop()
