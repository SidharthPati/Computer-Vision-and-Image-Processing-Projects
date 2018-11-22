# __author__ = 'spati'
# References
# Keypoint Detection : https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
# Keypoint matching : https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# Homography, RANSAC :
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Warp Perspective:
# https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545

import cv2
import numpy as np

# Reading both the mountain images
mountain_img1 = cv2.imread('mountain1.jpg')
mountain_img2 = cv2.imread('mountain2.jpg')

# Converting both the images to gray scale images
gray_scale_mountain_1 = cv2.cvtColor(mountain_img1, cv2.COLOR_BGR2GRAY)
gray_scale_mountain_2 = cv2.cvtColor(mountain_img2, cv2.COLOR_BGR2GRAY)

# Using SIFT to detect both the keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_mountain1 = sift.detect(gray_scale_mountain_1, None)
keypoints_mountain2 = sift.detect(gray_scale_mountain_2, None)

# Drawing the detected key points on both the images
# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS can be added for descriptors
keypoints_mountain1 = cv2.drawKeypoints(gray_scale_mountain_1, keypoints_mountain1, outImage=np.array([]),
                                        color=(0, 0, 0))
keypoints_mountain2 = cv2.drawKeypoints(gray_scale_mountain_2, keypoints_mountain2, outImage=np.array([]),
                                        color=(0, 0, 0))

# Writing both the images with keypoints on them
cv2.imwrite('task1_sift1.jpg', keypoints_mountain1)
cv2.imwrite('task1_sift2.jpg', keypoints_mountain2)

# find the keypoints and its descriptors using SIFT
keypoints_mountain1, descriptor1 = sift.detectAndCompute(mountain_img1, None)
keypoints_mountain2, descriptor2 = sift.detectAndCompute(mountain_img2, None)

# Match the keypoints using k-nearest neighbour using BFMatcher with default params
brute_force_matcher = cv2.BFMatcher()
matches = brute_force_matcher.knnMatch(descriptor1, descriptor2, k=2)

good_matches1 = []

for m, n in matches:
    if m.distance < 0.75*n.distance:
        good_matches1.append(m)
print("Printing good matches")
print(good_matches1)

draw_params = dict(matchColor=(0, 0, 255), singlePointColor=None, flags=2)
outImg = cv2.drawMatches(mountain_img1, keypoints_mountain1, mountain_img2, keypoints_mountain2,
                         good_matches1, None, **draw_params)

# Good matches
cv2.imwrite('task1_matches_knn.jpg', outImg)

# We set a condition that we should have at least 10 matches to find the object
src_pts = np.float32([keypoints_mountain1[m.queryIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_mountain2[m.trainIdx].pt for m in good_matches1]).reshape(-1, 1, 2)

homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

print("Printing Homography Matrix")
print(homography_matrix)

# Store the indices of all inliers in a new list
inliers_indices = []
for i in range(len(matchesMask)):
    if matchesMask[i] == 1:
        inliers_indices.append(i)

# Pick 10 random indices
random_inliers = np.random.choice(inliers_indices, 10)

# Fetch the good matches using indices
inliers_good_matches = []
for index in random_inliers:
    inliers_good_matches.append(good_matches1[index])

# matchesMask ensures only inliers are considered. Pick around 10 matches
random_matches_mask = list(np.random.choice(matchesMask, 10))
draw_params = dict(matchColor=(0, 0, 0), singlePointColor=None, flags=2)

img3 = cv2.drawMatches(mountain_img1, keypoints_mountain1, mountain_img2, keypoints_mountain2,
                       inliers_good_matches, None, **draw_params)

# Image with 10 inliers
cv2.imwrite('task1_matches.jpg', img3)
temp = mountain_img1
mountain_img1 = mountain_img2
mountain_img2 = temp
result = cv2.warpPerspective(mountain_img1, homography_matrix,
                             (mountain_img1.shape[1] + mountain_img2.shape[1], mountain_img1.shape[0]))
result[0:mountain_img2.shape[0], 0:mountain_img2.shape[1]] = mountain_img2
cv2.imwrite('result.jpg', result)

# Image warping
mountain_height1, mountain_weight1 = mountain_img1.shape[:2]
mountain_height2, mountain_weight2 = mountain_img2.shape[:2]
mount_pts1 = np.float32([[0, 0], [0, mountain_height1], [mountain_weight1, mountain_height1], [mountain_weight1, 0]]).reshape(-1, 1, 2)
mount_pts2 = np.float32([[0, 0], [0, mountain_height2], [mountain_weight2, mountain_height2], [mountain_weight2, 0]]).reshape(-1, 1, 2)
mount_transformed2 = cv2.perspectiveTransform(mount_pts2, homography_matrix)
pts = np.concatenate((mount_pts1, mount_transformed2), axis=0)

# Fetching the min and max points in x and y coordinates and translating
[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
trans = [-xmin, -ymin]
homo_trans = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])

# Final warping
result = cv2.warpPerspective(mountain_img2, homo_trans.dot(homography_matrix), (xmax-xmin, ymax-ymin))
result[trans[1]:mountain_height1+trans[1], trans[0]:mountain_weight1+trans[0]] = mountain_img1
cv2.imwrite('task1_pano.jpg', result)
