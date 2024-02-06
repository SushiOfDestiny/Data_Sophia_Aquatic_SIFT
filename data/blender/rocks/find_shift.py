import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread("left.png")
img2 = cv.imread("right.png")

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(img1)

axs[1].imshow(img2)

plt.show()
