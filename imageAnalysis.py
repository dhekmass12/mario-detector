"""
file: imageAnalysis.py
----------------------
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2

MIN_MATCH_COUNT = 10

def main(image1, image2, gray1, gray2, directory, verbose=True):
    # orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)
    orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if (kp1 is None) or (des1 is None) or (kp2 is None) or (des2 is None):
        return image2
    
    x = np.array([kp2[0].pt])

    for i in range(len(kp2)):
        x = np.append(x, [kp2[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)        
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    
    try:
        ms.fit(x)
    except:
        # print ("Bandwith param must be int the range (0.0 inf) or None.")
        return image2
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    # print("number of estimated clusters : %d" % n_clusters_)

    s = [None] * n_clusters_
    for i in range(n_clusters_):
        l = ms.labels_
        d, = np.where(l == i)
        # print(d.__len__())
        s[i] = list(kp2[xx] for xx in d)

    des2_ = des2

    for i in range(n_clusters_):
        kp2 = s[i]
        l = ms.labels_
        d, = np.where(l == i)
        des2 = des2_[d, ]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        des1 = np.float32(des1)
        des2 = np.float32(des2)

        matches = flann.knnMatch(des1, des2, 2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>=1:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            try:
                ## find homography matrix and do perspective transform
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except:
                # print ("Not enough keypoints/descriptions!")
                return image2

            if M is None:
                # print ("No Homography")
                return image2
            else:
                matchesMask = mask.ravel().tolist()

                h,w = gray1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                image2 = cv2.polylines(image2,[np.int32(dst)],True,255,2, cv2.LINE_AA)

                # image3 = cv2.drawMatches(gray1, kp1, gray2, kp2, good, None, flags=2)

        else:
            # print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            return image2
    
    return image2
