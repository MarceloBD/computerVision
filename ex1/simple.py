import cv2
import numpy as np
import time
start_time = time.time()

def paintArea(img, color):
	for i in xrange(len(img)):
		for j in xrange(len(img[0])):
			if(img[i][j][0] != 0 and img[i][j][1] != 0 and img[i][j][2] != 0):
				img[i][j] = color;


img = cv2.imread('cores.jpeg')
#cv2.imshow('img', img)
result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#k = cv2.waitKey(0) 
 # blue
lower = np.array([100,50,50])
upper = np.array([140,255,255])
mask = cv2.inRange(result, lower, upper)
blue = cv2.bitwise_and(img, img, mask= mask)
m, blue = cv2.threshold(blue, 10, 255, cv2.THRESH_BINARY)
bluecolor = np.array([255, 0, 0])
blue = cv2.bitwise_and(blue, bluecolor)

#paintArea(blue, [255, 0, 0])
# green
lower = np.array([40,50,50])
upper = np.array([60,255,255])
mask = cv2.inRange(result, lower, upper)
green = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((5,5),np.uint8)
#green = cv2.erode(green, kernel,iterations = 1)
m, green = cv2.threshold(green, 10, 255, cv2.THRESH_BINARY)
greencolor = np.array([0, 255, 0])
green = cv2.bitwise_and(green, greencolor)

#paintArea(green, [0, 255, 0])
# yellow
lower = np.array([10,140,50])
upper = np.array([40,255,255])
kernel = np.ones((10,10),np.uint8)
#yellow = cv2.erode(result, kernel,iterations = 1)
mask = cv2.inRange(result, lower, upper)
yellow = cv2.bitwise_and(img, img, mask= mask)
m, yellow = cv2.threshold(yellow, 10, 255, cv2.THRESH_BINARY)
yellowcolor = np.array([0, 255, 255])
yellow = cv2.bitwise_and(yellow, yellowcolor)

#paintArea(yellow, [0, 255, 255])
# orange 
lower = np.array([7,140,50])
upper = np.array([17,255,255])
kernel = np.ones((10,10),np.uint8)
#orange = cv2.erode(result, kernel,iterations = 1)
kernel = np.ones((5,5),np.uint8)
#orange = cv2.dilate(orange, kernel,iterations = 1)
mask = cv2.inRange(result, lower, upper)
orange = cv2.bitwise_and(img, img, mask= mask)
m, orange = cv2.threshold(orange, 10, 255, cv2.THRESH_BINARY)
orangecolor = np.array([0, 128, 255])
orange = cv2.bitwise_and(orange, orangecolor)

#paintArea(orange, [0, 128, 255])
# brown
lower = np.array([3,180,50])
upper = np.array([6,255,255])
kernel = np.ones((10,10),np.uint8)
#brown = cv2.erode(result, kernel,iterations = 1)
kernel = np.ones((5,5),np.uint8)
#brown = cv2.dilate(brown, kernel,iterations = 2)
mask = cv2.inRange(result, lower, upper)
brown = cv2.bitwise_and(img, img, mask= mask)
m, brown = cv2.threshold(brown, 10, 255, cv2.THRESH_BINARY)
browncolor = np.array([0, 76, 153])
brown = cv2.bitwise_and(brown, browncolor)
#paintArea(brow, [0, 76, 153])
# red
lower = np.array([3,50,150])
upper = np.array([7,255,255])
mask = cv2.inRange(result, lower, upper)
red = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((2,2),np.uint8)
#red = cv2.erode(red, kernel,iterations = 2)
kernel = np.ones((6,6),np.uint8)
#red = cv2.dilate(red, kernel,iterations = 2)
kernel = np.ones((7,7),np.uint8)
#red = cv2.erode(red, kernel,iterations = 2)
m, red = cv2.threshold(red, 10, 255, cv2.THRESH_BINARY)
redcolor = np.array([0, 0, 255])
red = cv2.bitwise_and(red, redcolor)

#paintArea(result, [0, 0, 255])

result = cv2.bitwise_or(brown, red)
result = cv2.bitwise_or(green, result)
result = cv2.bitwise_or(blue, result)
result = cv2.bitwise_or(yellow, result)
result = cv2.bitwise_or(orange, result)

#result = np.hstack((img, result))
#cv2.imshow('img', result)
#cv2.imwrite('simple.jpeg', result)
#k = cv2.waitKey(0) 
print("--- %s seconds ---" % (time.time() - start_time))


