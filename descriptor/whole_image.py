from descriptor import compute_descriptor_histograms2_rotated, compute_features_overall2
import cv2 as cv
import numpy as np

#compute_descriptor_vect = np.vectorize(compute_descriptor_histograms2_rotated)

if __name__ == '__main__':
    g_img = cv.imread("../data/blue_cylinder.jpg", cv.IMREAD_GRAYSCALE).astype(np.float64)
    features = compute_features_overall2(g_img)
    des = compute_descriptor_histograms2_rotated(features, (0, 0))
    n, m = np.shape(g_img)
    descriptors = np.zeros([n, m, 3, np.shape(des[0])[0], np.shape(des[0])[1], np.shape(des[0])[2]])
    for i in range(n):
        for j in range(m):
            des = compute_descriptor_histograms2_rotated(features, (i, j))
            print("---")
            print("Descriptor calculated for point "+str(i*n+j)+" out of "+str(n*m))
            descriptors[i, j, 0] = des[0]
            descriptors[i, j, 1] = des[1]
            descriptors[i, j, 2] = des[2]