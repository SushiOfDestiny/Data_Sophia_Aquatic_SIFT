import matplotlib.pyplot as plt
from filenames_creation import original_imgs_path_prefix, matches_path
from computation_pipeline_hyper_params import im_name1
import numpy as np

if __name__ == '__main__':
    im1 = plt.imread(original_imgs_path_prefix+'/'+im_name1+'.png')
    dmap = np.load(matches_path+'/dmap.npy')
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im1)
    dmap_plot = ax[1].imshow(dmap, cmap='magma')
    plt.colorbar(dmap_plot, ax=ax[1])
    plt.show()