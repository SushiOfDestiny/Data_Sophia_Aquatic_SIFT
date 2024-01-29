import cv2
import pickle
import numpy as np

def save_keypoints(kp, filename):
    '''Save OpenCV keypoints to file'''
    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
        index.append(temp)
    f = open(filename, 'wb')
    try:
        f.write(pickle.dumps(index))
    finally:
        f.close()

def save_matches(matches, filename):
    '''Save OpenCV matches to file'''
    print(matches)
    index = []
    for match in matches:
        temp = [(match[0].distance, match[0].trainIdx, match[0].queryIdx, match[0].imgIdx)]
        index.append(temp)
    f = open(filename, 'wb')
    try:
        f.write(pickle.dumps(index))
    finally:
        f.close()

def load_matches(filename):
    with open(filename, 'rb') as f:
        index = pickle.loads(f.read())
    matches = []
    for match in index:
        temp = [cv2.DMatch(_distance=match[0][0], _trainIdx=match[0][1], _queryIdx=match[0][2], _imgIdx=match[0][3])]
        matches.append(temp)
    return matches

def load_keypoints(filename):
    with open(filename, 'rb') as f:
        index = pickle.loads(f.read())
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1], size=point[1], angle=point[2], 
                                response=point[3], octave=point[4], class_id=point[5]) 
        kp.append(temp)
    return kp

def get_pts_coords_from_pt_pair(pt_pair):
    # Auxiliary function
    return ((pt_pair[0].pt[0], pt_pair[0].pt[1]), (pt_pair[1].pt[0], pt_pair[1].pt[1]))

def save_kp_pairs_to_arr(kp_pairs, file):
    '''Saves a list of OpenCV keypoint pairs to a numpy array file
    
    Returns :
    - None'''
    kp_arr = np.empty((2, len(kp_pairs), 2), dtype=np.float64)
    for i, kp_pair in enumerate(kp_pairs):
        pt1, pt2 = get_pts_coords_from_pt_pair(kp_pair)
        kp_arr[0, i, 0] = pt1[0]
        kp_arr[0, i, 1] = pt1[1]
        kp_arr[1, i, 0] = pt2[0]
        kp_arr[1, i, 1] = pt2[1]
    np.save(file, kp_arr)