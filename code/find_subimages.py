import cv2 as cv
import matplotlib.pyplot as plt
from computation_pipeline_hyper_params import *
from filenames_creation import *

# Goal is to find matching subimages in the 2 images, to compute descriptors on them afterwards

if __name__ == "__main__":
    # Load images
    ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for id_image in range(2):
        axs[id_image].imshow(ims[id_image], cmap="gray")

    plt.show()
