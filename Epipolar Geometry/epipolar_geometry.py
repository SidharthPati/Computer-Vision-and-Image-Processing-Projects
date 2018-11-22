# __author__ = 'spati'
# References
# Fundamental Matrix:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
# Disparity map: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
# Stereo SGBM: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html

import cv2
import numpy as np

# Reading both the sculpture images
tsucuba_left = cv2.imread('tsucuba_left.png')
tsucuba_right = cv2.imread('tsucuba_right.png')

# Converting both the images to gray scale images
gray_scale_tsucuba_left = cv2.cvtColor(tsucuba_left, cv2.COLOR_BGR2GRAY)
gray_scale_tsucuba_right = cv2.cvtColor(tsucuba_right, cv2.COLOR_BGR2GRAY)

# Using SIFT to detect both the keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_tsucuba_left = sift.detect(gray_scale_tsucuba_left, None)
keypoints_tsucuba_right = sift.detect(gray_scale_tsucuba_right, None)

# Drawing the detected key points on both the images
# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS can be added for descriptors
keypoints_tsucuba_left = cv2.drawKeypoints(gray_scale_tsucuba_left, keypoints_tsucuba_left, outImage=np.array([]),
                                        color=(0, 0, 0))
keypoints_tsucuba_right = cv2.drawKeypoints(gray_scale_tsucuba_right, keypoints_tsucuba_right, outImage=np.array([]),
                                        color=(0, 0, 0))

# Task 2.1 output
# Writing both the images with keypoints on them
cv2.imwrite('task2_sift1.jpg', keypoints_tsucuba_left)
cv2.imwrite('task2_sift2.jpg', keypoints_tsucuba_right)

# find the keypoints and its descriptors using SIFT
keypoints_tsucuba_left, descriptor1 = sift.detectAndCompute(tsucuba_left, None)
keypoints_tsucuba_right, descriptor2 = sift.detectAndCompute(tsucuba_right, None)

# Match the keypoints using k-nearest neighbour using BFMatcher with default params
brute_force_matcher = cv2.BFMatcher()
matches = brute_force_matcher.knnMatch(descriptor1, descriptor2, k=2)

good_matches = []
good_matches1 = []
left_points = []
right_points = []

for m, n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])
        good_matches1.append(m)
        right_points.append(keypoints_tsucuba_right[m.trainIdx].pt)
        left_points.append(keypoints_tsucuba_left[m.queryIdx].pt)

# cv2.drawMatchesKnn expects list of lists as matches
outImg = np.array([])
outImg = cv2.drawMatchesKnn(tsucuba_left,
                            keypoints_tsucuba_left,
                            tsucuba_right,
                            keypoints_tsucuba_right,
                            good_matches,
                            outImg,
                            flags=2)

# Task 1.1 output
cv2.imwrite('task2_matches_knn.jpg', outImg)

# Finding Fundamental Matrix
left_points = np.int32(left_points)
right_points = np.int32(right_points)
fundamental_matrix, mask = cv2.findFundamentalMat(left_points, right_points, cv2.RANSAC)

# We select only inlier points
left_points = left_points[mask.ravel() == 1]
right_points = right_points[mask.ravel() == 1]
print("left points:")
print(len(left_points))
print("right points:")
print(len(right_points))

# Pick 10 random indices
# np.random.seed(sum([ord(c) for c in UBIT]))
random_inliers_indices = np.random.choice(range(len(left_points)), 11)
left_inliers = []
right_inliers = []

for index in random_inliers_indices:
    left_inliers.append(left_points[index])
    right_inliers.append(right_points[index])

# Task 2.2 output
print("Fundamental Matrix: ")
print(fundamental_matrix)


# function to draw lines
def drawlines(img1, img2, lines, pts1, pts2, colors):
    """

    :param img1: 1st image
    :param img2: 2nd Image
    :param lines: epipolar lines
    :param pts1: points on 1st image
    :param pts2: points on 2nd image
    :param colors: Tuple of colors to be used to draw lines
    :return:
    """
    row, column, unwanted = img1.shape
    for row, pt1, pt2, color in zip(lines, pts1, pts2, colors):
        x0, y0 = map(int, [0, -row[2]/row[1]])
        x1, y1 = map(int, [column, -(row[2]+row[0]*column)/row[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
    return img1, img2


color = [[0 for k in range(3)] for j in range(11)]
for i in range(11):
    color[i] = tuple(np.random.randint(0,255,3).tolist())

left_lines = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, fundamental_matrix)
left_lines = left_lines.reshape(-1, 3)
final_left, img1 = drawlines(tsucuba_left, tsucuba_right, left_lines, left_inliers, right_inliers, color)

right_lines = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 2, fundamental_matrix)
right_lines = right_lines.reshape(-1, 3)
final_right, img1 = drawlines(tsucuba_right, tsucuba_left, right_lines, right_inliers, left_inliers, color)

# Task 2.3 output
cv2.imwrite('task2_epi_left.jpg', final_left)
cv2.imwrite('task2_epi_right.jpg', final_right)

# Disparity Map
window_size = 2
min_disp = 16
num_disp = 112 - min_disp

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=16,
                               P1=8*3*window_size**2,
                               P2=32*3*window_size**2)

disparity = stereo.compute(gray_scale_tsucuba_left, gray_scale_tsucuba_right).astype(np.float32)/16

# final disparity map
cv2.imwrite('task2_disparity.jpg', 255*(disparity-min_disp)/num_disp)
