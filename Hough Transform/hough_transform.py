# __author__ = 'spati'
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
import cv2
import numpy as np
import math


# 3*3 Sobel Operator for horizontal edges
def sobel_x():
    """
    Sobel filter in x direction
    :return: Sobel Filter for x direction
    """
    # Creates a list containing h lists, each of w items, all set to 0
    w, h = 3, 3
    S_x = [[0 for x in range(w)] for y in range(h)]
    # Add values to the sobel filter
    S_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    S_x_flipped = flip_sobel(S_x)
    return S_x_flipped


# 3*3 Sobel Operator for vertical edges
def sobel_y():
    """
    Sobel filter in y direction
    :return: Sobel Filter for y direction
    """
    # Creates a list containing h lists, each of w items, all set to 0
    w, h = 3, 3
    S_y = [[0 for x in range(w)] for y in range(h)]
    # Add values to the sobel filter
    S_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    S_y_flipped = flip_sobel(S_y)
    return S_y_flipped


def flip_sobel(matrix):
    """
    returns list after rotating the 3*3 sobel filter horizontally and vertically
    :param matrix: Sobel filter in either x or y direction
    :return: Flipped Sobel Filter
    """
    for i in range(0, 3):
        temp = matrix[i][0]
        matrix[i][0] = matrix[i][2]
        matrix[i][2] = temp
    for j in range(0, 3):
        temp = matrix[0][j]
        matrix[0][j] = matrix[2][j]
        matrix[2][j] = temp
    return matrix


def convolution(image_matrix, sobel_filter):
    """
    :param image_matrix: Image to be convoluted with the sobel filter
    :param sobel_filter: Sobel Filter in either x or y direction
    :return: final image after convolution
    """
    # create the new image
    w1, h1 = image_matrix.shape[1]-2, image_matrix.shape[0]-2
    new_image_matrix = np.asarray([[0 for x in range(w1)] for y in range(h1)])
    min_value = max_value = 0  # Use this for normalization
    for k in range(1, h1+1):
        for l in range(1, w1+1):
            sum1 = 0

            sum1 = sum1 + (sobel_filter[0][0]*image_matrix[k-1][l-1] +
                           sobel_filter[0][1]*image_matrix[k-1][l] +
                           sobel_filter[0][2]*image_matrix[k-1][l+1] +
                           sobel_filter[1][0] * image_matrix[k][l-1] +
                           sobel_filter[1][1]*image_matrix[k][l] +
                           sobel_filter[1][2]*image_matrix[k][l+1] +
                           sobel_filter[2][0]*image_matrix[k+1][l-1] +
                           sobel_filter[2][1]*image_matrix[k+1][l] +
                           sobel_filter[2][2]*image_matrix[k+1][l+1])
            if abs(sum1) < min_value:
                min_value = sum1
            if abs(sum1) > max_value:
                max_value = sum1

            new_image_matrix[k-1][l-1] = sum1
    return new_image_matrix, max_value


def combine_xandy(x_matrix, y_matrix):
    """
    :param x_matrix: Image after applying Sobel in x direction
    :param y_matrix: Image after applying Sobel in y direction
    :return: Image containing horizontal and vertical edges
    """
    row_matrix = []
    combined_matrix = []

    for x in range(len(x_matrix)):
        for y in range(len(x_matrix[0])):
            magnitude = math.sqrt(x_matrix[x][y]**2 + y_matrix[x][y]**2)
            row_matrix.append(magnitude)
        combined_matrix.append(row_matrix)
        row_matrix = []
    return combined_matrix


def thresholding(threshold, matrix):
    """
    Thresholds the matrix values. Anything below the threshold is 0 and anything above it is 255.
    :param threshold: threshold value
    :param matrix: image matrix
    :return: resultant image matrix after thresholding
    """
    w1, h1 = matrix.shape[1], matrix.shape[0]
    for x in range(h1):
        for y in range(w1):
            if matrix[x][y] > threshold:
                matrix[x][y] = 255
            elif matrix[x][y] <= threshold:
                matrix[x][y] = 0
    return matrix


def accumulator_matrix(matrix, org_img, red_img, blue_img):
    """
    Determine accumulator matrix for lines and draw lines on images
    :param matrix: Edge detected image
    :param org_img: Original image on which lines are drawn
    :param red_img: Original image on which red lines are drawn
    :param blue_img: Original image on which blue lines are drawn
    """
    # calculating rho and theta values
    d_list = []
    w1, h1 = matrix.shape[1], matrix.shape[0]
    for x in range(h1):
        for y in range(w1):
            if matrix[x][y] == 255:
                for theta in range(0, 181):
                    d = int(x * (math.cos(math.radians(theta))) - y * (math.sin(math.radians(theta))))
                    d_list.append([d, theta])

    d_list = np.asarray(d_list)
    d_min = np.min(d_list[:, 0])

    # getting rid of negative values
    for i in range(len(d_list)):
        d_list[i][0] += abs(d_min)

    # find the max value
    d_max = int(np.max(d_list[:, 0]))

    # new accumulator array
    accumulator_matrix_new = np.zeros((d_max+1, 181))

    # voting for lines
    for i in range(len(d_list)):
        d_val = int(d_list[i][0])
        d_theta = int(d_list[i][1])
        accumulator_matrix_new[d_val, d_theta] += 1

    # sinusoidal waves obtained in hough space
    cv2.imwrite('sinusoidal.jpg', np.asarray(accumulator_matrix_new, dtype='uint8'))

    # pick the max vote
    max_vote = np.max(accumulator_matrix_new)

    print(max_vote)

    vote_threshold = 0.27 * max_vote  # change this

    max_votes_list = []

    for d in range(d_max):
        for theta in range(181):
            if accumulator_matrix_new[d][theta] > vote_threshold:
                max_votes_list.append([d, theta])

    print(max_votes_list)

    # loop to detect all lines
    for pair in max_votes_list:
        d = pair[0] - abs(d_min)
        theta = pair[1]
        if theta == 90:
            continue
        y1 = 0
        y2 = 1000

        x1 = int((y1 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) + (d / math.cos(math.radians(theta))))
        x2 = int((y2 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) + (d / math.cos(math.radians(theta))))
        cv2.line(org_img, (y1, x1), (y2, x2), (0, 0, 255), 2)

    # loop to detect red lines
    for pair in max_votes_list:
        d = pair[0] - abs(d_min)
        theta = pair[1]
        if theta in [85, 86, 87, 88, 89]:
            y1 = 0
            y2 = 1000

            x1 = int((y1 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) + (d / math.cos(math.radians(theta))))
            x2 = int((y2 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) + (d / math.cos(math.radians(theta))))
            cv2.line(red_img, (y1, x1), (y2, x2), (0, 0, 255), 2)

    # loop to detect blue lines
    for pair in max_votes_list:
        d = pair[0] - abs(d_min)
        theta = pair[1]
        if theta in [53, 54, 55, 56]:
            y1 = 0
            y2 = 1000

            x1 = int((y1 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) +
                     (d / math.cos(math.radians(theta))))
            x2 = int((y2 * math.sin(math.radians(theta)) / math.cos(math.radians(theta))) +
                     (d / math.cos(math.radians(theta))))
            cv2.line(blue_img, (y1, x1), (y2, x2), (255, 0, 255), 2)


def circle_accumulator_matrix(matrix, circle_img):
    """
    Arbitrary matrix for circle. We're using estimated radius of circles to detect
    :param matrix: edge detected image
    :param circle_img: Image with circles detected
    :return: image with circles detected
    """
    voting_list = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 255:
                for r in [20, 21, 22, 23, 24]:  # estimated radius
                    for theta in range(361):
                        a = int(j - r * math.cos(math.radians(theta)))
                        b = int(i + r * math.sin(math.radians(theta)))
                        voting_list.append((a, b, (r - 20)))

    # find maximum of center's coordinates a,b and radius, r
    voting_list = np.asarray(voting_list)

    # creating outer limits for ranges
    a = np.max(voting_list[:, 0]) + 1
    b = np.max(voting_list[:, 1]) + 1
    r = np.max(voting_list[:, 2]) + 1

    # create new accumulator matrix
    acc_matrix_circle = np.zeros((a, b, r))

    # Voting
    for i in range(len(voting_list)):
        if (voting_list[i][0] >= 0) and (voting_list[i][1] >= 0):
            acc_matrix_circle[voting_list[i][0], voting_list[i][1], voting_list[i][2]] += 1

    max_value = int(np.max(acc_matrix_circle))

    # finding max points which are greater than the threshold
    max_points_array = []
    threshold_center = max_value * 0.6
    for i in range(a):
        for j in range(b):
            for k in range(r):
                if acc_matrix_circle[i, j, k] > threshold_center:
                    max_points_array.append((i, j, k))

    # draw circles
    for point in max_points_array:
        cv2.circle(circle_img, (point[0], point[1]), (point[2] + 20), (255, 0, 0), 1)


# read the image
img = cv2.imread("hough.jpg", 0)

# final images
final_image_copy = cv2.imread("hough.jpg")
final_red_lines = cv2.imread("hough.jpg")
final_blue_lines = cv2.imread("hough.jpg")
final_circle_image = cv2.imread("hough.jpg")

# convert to np array
img = np.asarray(img)

# pad 0s on the boundaries
w, h = img.shape[1]+2, img.shape[0]+2
padded_img = np.asarray([[0 for x in range(w)] for y in range(h)])

# combine original image to the padded image
padded_img[1:-1, 1:-1] = img

# calculate gradients along x and y axis
gradient_x = sobel_x()
gradient_y = sobel_y()

# convolution along x and y
final_image_matrix, max_x_value = convolution(padded_img, gradient_x)
final_image_matrix1, max_y_value = convolution(padded_img, gradient_y)

# combine both x and y
final_combined_image = combine_xandy(final_image_matrix, final_image_matrix1)

# find the max value and divide it for
max_val = np.max(final_combined_image)
final_combined_image = final_combined_image/max_val
final_combined_image = final_combined_image*255

# thresholding after detecting edges
threshold_val = 40
final_combined_image = thresholding(threshold_val, final_combined_image)
final_combined_image = np.asarray(final_combined_image)

# save the image with the edges detected
cv2.imwrite('edges_detected.jpg', np.asarray(final_combined_image, dtype='uint8'))

# calculate accumulator matrix and detect red and blue lines
accumulator_matrix(final_combined_image, final_image_copy, final_red_lines, final_blue_lines)

# writing images after detecting lines
cv2.imwrite('final_image.jpg', np.asarray(final_image_copy, dtype='uint8'))
cv2.imwrite('red_line.jpg', np.asarray(final_red_lines, dtype='uint8'))
cv2.imwrite('blue_lines.jpg', np.asarray(final_blue_lines, dtype='uint8'))

# code for detecting circles
circle_accumulator_matrix(final_combined_image, final_circle_image)

# writing the image after detecting circles
cv2.imwrite('coin.jpg', np.asarray(final_circle_image, dtype='uint8'))
