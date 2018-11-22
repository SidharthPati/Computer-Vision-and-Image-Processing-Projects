# __author__ = 'spati'
import cv2
import numpy as np
import sys
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
    S_x[0][0] = 1
    S_x[0][1] = 0
    S_x[0][2] = -1
    S_x[1][0] = 2
    S_x[1][1] = 0
    S_x[1][2] = -2
    S_x[2][0] = 1
    S_x[2][1] = 0
    S_x[2][2] = -1
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
    S_y[0][0] = 1
    S_y[0][1] = 2
    S_y[0][2] = 1
    S_y[1][0] = 0
    S_y[1][1] = 0
    S_y[1][2] = 0
    S_y[2][0] = -1
    S_y[2][1] = -2
    S_y[2][2] = -1
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
    w1, h1 = 900, 600
    new_image_matrix = np.asarray([[0 for x in range(w1)] for y in range(h1)])
    min_value = max_value = 0  # Use this for normalization
    for k in range(1, 601):
        for l in range(1, 901):
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


def normalize(image, max_value):
    """

    :param image: Final image
    :param max_value: maximum value of pixel in image
    :return: Image after normalization
    """
    for k in range(0, 600):
        for l in range(0, 900):
            image[k][l] = abs(image[k][l])/abs(max_value)
    return image


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


np.set_printoptions(threshold=sys.maxsize)

# read the image
img = cv2.imread("task1.png", 0)
# convert to np array
img = np.asarray(img)

# pad 0s
w, h = 902, 602
padded_img = np.asarray([[0 for x in range(w)] for y in range(h)])

# combine original image to the padded image
padded_img[1:-1, 1:-1] = img
cv2.imshow('edge_direction4', np.asarray(padded_img, dtype='uint8'))

gradient_x = sobel_x()
gradient_y = sobel_y()

final_image_matrix, max_x_value = convolution(padded_img, gradient_x)
final_image_matrix1, max_y_value = convolution(padded_img, gradient_y)

# final_image_matrix = normalize(final_image_matrix, max_x_value)
# final_image_matrix1 = normalize(final_image_matrix1, max_y_value)

final_combined_image = combine_xandy(final_image_matrix, final_image_matrix1)
final_combined_image = np.asarray(final_combined_image)

print("Original image size: ")
print(img.shape)
print("Resulting image size: ")
print(final_image_matrix.shape)

cv2.imshow('edge_direction', np.asarray(final_image_matrix, dtype='uint8'))
cv2.imshow('edge_direction1', np.asarray(final_image_matrix1, dtype='uint8'))
cv2.imshow('final_image', np.asarray(final_combined_image, dtype='uint8'))
cv2.imwrite('edges_detected.jpg', np.asarray(final_combined_image, dtype='uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
