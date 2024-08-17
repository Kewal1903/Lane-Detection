import cv2 as cv
import numpy as np

image = cv.imread("C:\\Users\\Kewal\\Desktop\\open-road-kimberly-western-australia-260nw-1589764003.webp")
img = image[0:258,:]

cv.imshow("Original Image",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale image",gray)

sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)
edges = np.sqrt(sobelx**2+sobely**2)

edges = np.uint8(edges / np.max(edges) * 255)

edges = cv.threshold(edges,50,255,cv.THRESH_BINARY)[1]

cv.imshow("Edge-detected Image",edges)

lines = cv.HoughLinesP(edges,1, np.pi / 180,100,minLineLength=20,maxLineGap=0)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img,(x1, y1),(x2, y2),(0, 255, 0),2)


cv.imshow("Detected Lane", img)
cv.waitKey(0)
