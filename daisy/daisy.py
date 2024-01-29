from skimage.feature import daisy
from skimage import data, color, io

import matplotlib.pyplot as plt

img = data.camera()
print(img.shape)
img1 = color.rgb2gray(io.imread("data/irl/scene_zipped/scene_l_1_u.jpeg"))
descs, descs_img = daisy(img1, step=200, radius=100, rings=2, histograms=8, orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(descs_img)
plt.show()