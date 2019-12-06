import numpy as np
from numpy.linalg import norm
from scipy import misc
from scipy.ndimage import gaussian_filter
import cv2


def get_sift_keypoints(image):
    sift_cv2 = cv2.xfeatures2d.SIFT_create()
    keypoints = sift_cv2.detect(image, None)
    return keypoints


def stitch_images(input_left, input_right):
    status, H = match(input_left.copy(), input_right.copy())

    if status:
        outputImage = cv2.warpPerspective(input_right, H, (input_left.shape[1] + input_right.shape[1], input_left.shape[0]))
        outputImage[0:input_left.shape[0], 0:input_left.shape[1]] = input_left
    return True, outputImage


def match(image_left, image_right):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    sift = cv2.xfeatures2d.SIFT_create()

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    keypoints_left, descriptors_left = sift.detectAndCompute(image_left_gray, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(image_right_gray, None)

    matches = flann.knnMatch(descriptors_right, descriptors_left, k=2)

    # Ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
        pointsCurrent = keypoints_right
        pointsPrevious = keypoints_left

        matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
        matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])

        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
        return True, H
    else:
        return False, None
