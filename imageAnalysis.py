"""
file: imageAnalysis.py
----------------------
"""

import numpy as np
import cv2

def main(image1, image2, directory, verbose=True):
    ## Create ORB object and BF object(using HAMMING)
    orb = cv2.ORB_create()
    
    image2_copy = image2.copy()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    LIMIT = 5
    while (LIMIT > 0):
        gray2 = cv2.cvtColor(image2_copy, cv2.COLOR_BGR2GRAY)
        
        ## Find the keypoints and descriptors with ORB
        kpts1, descs1 = orb.detectAndCompute(gray1,None)
        kpts2, descs2 = orb.detectAndCompute(gray2,None)
        
        if (kpts1 is None) or (kpts2 is None) or (descs1 is None) or (descs2 is None):
            return image2

        ## match descriptors and sort them in the order of their distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descs1, descs2)
        dmatches = sorted(matches, key = lambda x:x.distance)

        ## extract the matched keypoints
        src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

        try:
            ## find homography matrix and do perspective transform
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            return image2
            
            
        try:
            h,w = image1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            ## draw found regions
            image2 = cv2.polylines(image2, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)
                
            # roi_corners = np.array([[(100,100),(100, 800),(800, 800),(800, 100)]],dtype = np.int32)
            # blurred_image = cv2.blur(image2_copy, (100, 100))
            # mask = np.zeros(image2_copy.shape, dtype=np.uint8)
            # channel_count = image2_copy.shape[2]
            # ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(image2_copy, [np.int32(dst)], (180, 177, 171))
            # mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
            # image2_copy = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image2_copy, mask_inverse)
            
            ## draw match lines
            # res = cv2.drawMatches(image1, kpts1, image2, kpts2, dmatches[:20],None,flags=2)
        except:
            pass
        
        LIMIT -= 1
        
    return image2