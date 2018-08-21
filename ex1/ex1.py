import cv2
import numpy as np


def paintArea(img, color):
	for i in xrange(len(img)):
		for j in xrange(len(img[0])):
			if(img[i][j][0] != 0 and img[i][j][1] != 0 and img[i][j][2] != 0):
				img[i][j] = color;


img = cv2.imread('cores.jpeg')
cv2.imshow('img', img)
result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


k = cv2.waitKey(0) 
 # blue
lower = np.array([100,50,50])
upper = np.array([140,255,255])
mask = cv2.inRange(result, lower, upper)
blue = cv2.bitwise_and(img, img, mask= mask)
paintArea(blue, [255, 0, 0])
# green
lower = np.array([40,50,50])
upper = np.array([60,255,255])
mask = cv2.inRange(result, lower, upper)
green = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((5,5),np.uint8)
green = cv2.erode(green, kernel,iterations = 1)
paintArea(green, [0, 255, 0])
# yellow
lower = np.array([10,140,50])
upper = np.array([40,255,255])
kernel = np.ones((10,10),np.uint8)
yellow = cv2.erode(result, kernel,iterations = 1)
mask = cv2.inRange(yellow, lower, upper)
yellow = cv2.bitwise_and(img, img, mask= mask)
paintArea(yellow, [0, 255, 255])
# orange 
lower = np.array([7,140,50])
upper = np.array([17,255,255])
kernel = np.ones((10,10),np.uint8)
orange = cv2.erode(result, kernel,iterations = 1)
mask = cv2.inRange(orange, lower, upper)
orange = cv2.bitwise_and(img, img, mask= mask)
paintArea(orange, [0, 128, 255])
# brown
lower = np.array([3,180,50])
upper = np.array([6,255,255])
kernel = np.ones((10,10),np.uint8)
brow = cv2.erode(result, kernel,iterations = 1)
mask = cv2.inRange(brow, lower, upper)
brow = cv2.bitwise_and(img, img, mask= mask)
paintArea(brow, [0, 76, 153])
# red
lower = np.array([3,50,150])
upper = np.array([7,255,255])
mask = cv2.inRange(result, lower, upper)
result = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((2,2),np.uint8)
result = cv2.erode(result, kernel,iterations = 2)
kernel = np.ones((6,6),np.uint8)
result = cv2.dilate(result, kernel,iterations = 2)
kernel = np.ones((7,7),np.uint8)
result = cv2.erode(result, kernel,iterations = 2)
paintArea(result, [0, 0, 255])

result = cv2.bitwise_or(brow, result)
result = cv2.bitwise_or(green, result)
result = cv2.bitwise_or(blue, result)
result = cv2.bitwise_or(yellow, result)
result = cv2.bitwise_or(orange, result)

cv2.imshow('img', result)
k = cv2.waitKey(0) 



