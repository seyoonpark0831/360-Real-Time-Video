from threading import Thread
import cv2
from PyQt4 import QtCore, QtGui

class WebcamVideoStream:
	def __init__(self, src):
		self.stream = cv2.VideoCapture(src)
		self.stream.set(3,1280)
		self.stream.set(4,720)
		self.stream.set(5,30)

		(self.grabbed, self.frame) = self.stream.read()

		self.stopped = False

	def start(self):
		self._start = datetime.datetime.now()
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while True:
			QtGui.QApplication.processEvents()
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True
