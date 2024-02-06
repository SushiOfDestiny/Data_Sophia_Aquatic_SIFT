import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('left.png')
img2 = cv.imread('right.png')

plt.figure(0)
plt.imshow(img1)
plt.figure(1)
plt.imshow(img2)
plt.show()