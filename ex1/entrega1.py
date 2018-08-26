import cv2
import numpy as np

img = cv2.imread('cores.jpeg')
result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

 # blue
lower = np.array([100,50,50])
upper = np.array([140,255,255])
mask = cv2.inRange(result, lower, upper)
blue = cv2.bitwise_and(img, img, mask= mask)
m, blue = cv2.threshold(blue, 10, 255, cv2.THRESH_BINARY)
bluecolor = np.array([255, 0, 0])
blue = cv2.bitwise_and(blue, bluecolor)

# green
lower = np.array([40,50,50])
upper = np.array([60,255,255])
mask = cv2.inRange(result, lower, upper)
green = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((5,5),np.uint8)
green = cv2.erode(green, kernel,iterations = 1)
m, green = cv2.threshold(green, 10, 255, cv2.THRESH_BINARY)
greencolor = np.array([0, 255, 0])
green = cv2.bitwise_and(green, greencolor)

# yellow
lower = np.array([10,140,50])
upper = np.array([40,255,255])
kernel = np.ones((10,10),np.uint8)
yellow = cv2.erode(result, kernel,iterations = 1)
mask = cv2.inRange(yellow, lower, upper)
yellow = cv2.bitwise_and(img, img, mask= mask)
m, yellow = cv2.threshold(yellow, 10, 255, cv2.THRESH_BINARY)
yellowcolor = np.array([0, 255, 255])
yellow = cv2.bitwise_and(yellow, yellowcolor)

# orange 
lower = np.array([7,140,50])
upper = np.array([17,255,255])
kernel = np.ones((10,10),np.uint8)
orange = cv2.erode(result, kernel,iterations = 1)
kernel = np.ones((5,5),np.uint8)
orange = cv2.dilate(orange, kernel,iterations = 1)
mask = cv2.inRange(orange, lower, upper)
orange = cv2.bitwise_and(img, img, mask= mask)
m, orange = cv2.threshold(orange, 10, 255, cv2.THRESH_BINARY)
orangecolor = np.array([0, 128, 255])
orange = cv2.bitwise_and(orange, orangecolor)

# brown
lower = np.array([3,180,50])
upper = np.array([6,255,255])
kernel = np.ones((10,10),np.uint8)
brown = cv2.erode(result, kernel,iterations = 1)
kernel = np.ones((5,5),np.uint8)
brown = cv2.dilate(brown, kernel,iterations = 2)
mask = cv2.inRange(brown, lower, upper)
brown = cv2.bitwise_and(img, img, mask= mask)
m, brown = cv2.threshold(brown, 10, 255, cv2.THRESH_BINARY)
browncolor = np.array([0, 76, 153])
brown = cv2.bitwise_and(brown, browncolor)

# red
lower = np.array([3,50,150])
upper = np.array([7,255,255])
mask = cv2.inRange(result, lower, upper)
red = cv2.bitwise_and(img, img, mask= mask)
kernel = np.ones((2,2),np.uint8)
red = cv2.erode(red, kernel,iterations = 2)
kernel = np.ones((6,6),np.uint8)
red = cv2.dilate(red, kernel,iterations = 2)
kernel = np.ones((7,7),np.uint8)
red = cv2.erode(red, kernel,iterations = 2)
m, red = cv2.threshold(red, 10, 255, cv2.THRESH_BINARY)
redcolor = np.array([0, 0, 255])
red = cv2.bitwise_and(red, redcolor)

result = cv2.bitwise_or(brown, red)
result = cv2.bitwise_or(green, result)
result = cv2.bitwise_or(blue, result)
result = cv2.bitwise_or(yellow, result)
result = cv2.bitwise_or(orange, result)

cv2.imwrite('final.jpeg', result)