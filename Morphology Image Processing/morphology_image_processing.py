# __author__ = 'spati'
import cv2
import numpy as np


def dilation(img, struct_element):
    """

    :param img: padded image
    :param struct_element: structuring element
    :return: dilated image
    """
    width_img = len(img[0])
    height_img = len(img)
    dilated_img = np.asarray([[0 for x in range(width_img)] for y in range(height_img)])
    for i in range(1, height_img-1):
        for j in range(1, width_img-1):
            struct_x = 1
            struct_y = 1
            for k in range(-1, 1):
                for l in range(-1, 1):
                    if img[i+k][j+l] == 255 and struct_element[struct_x+k][struct_y+l] == 1:
                        dilated_img[i][j] = 255
    return dilated_img


def erosion(img, struct_element, size=None):
    """

    :param img: padded image
    :param struct_element: structuring element
    :return: eroded image
    """
    width_img = len(img[0])
    height_img = len(img)
    eroded_img = np.asarray([[0 for x in range(width_img)] for y in range(height_img)])
    for i in range(1, height_img-1):
        for j in range(1, width_img-1):
            struct_x = 1
            struct_y = 1
            count_struct = 0
            count_img = 0
            if size == 5:
                for k in range(-2, 2):
                    for l in range(-2, 2):
                        if struct_element[struct_x + k][struct_y + l] == 1:
                            count_struct += 1
                            if img[i+k][j+l] == 255:
                                count_img += 1
            else:
                for k in range(-1, 1):
                    for l in range(-1, 1):
                        if struct_element[struct_x + k][struct_y + l] == 1:
                            count_struct += 1
                            if img[i+k][j+l] == 255:
                                count_img += 1
            if count_struct > 0 and count_struct == count_img:
                eroded_img[i][j] = 255
    return eroded_img


def opening(img, struct_element):
    """
    Performs Erosion and then Dilation
    :param img: padded image
    :param struct_element: structuring element
    :return: Image after performing Opening operation
    """
    eroded_img = erosion(img, struct_element)
    opening_img = dilation(eroded_img, struct_element)
    return opening_img


def closing(img, struct_element):
    """
    Performs Dilation and then Erosion
    :param img: padded image
    :param struct_element: structuring element
    :return: Image after performing Closing operation
    """
    dilated_img = dilation(img, struct_element)
    closing_img = erosion(dilated_img, struct_element)
    return closing_img


# Reading the image
noise_img = cv2.imread('noise.jpg', 0)
structuring_element = [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]
# structuring element to fetch boundaries
structuring_element_5 = [[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]]

# convert image to np array
np_img = np.asarray(noise_img)

# pad 0s
w, h = np_img.shape[1]+2, np_img.shape[0]+2
padded_img = np.asarray([[0 for x in range(w)] for y in range(h)])

# combine original image to the padded image
padded_img[1:-1, 1:-1] = np_img

# dilation
ret_img = dilation(padded_img, structuring_element)
cv2.imwrite('some_dilated_img.jpg', np.asarray(ret_img, dtype='uint8'))

# erosion
ret_img1 = erosion(padded_img, structuring_element)
cv2.imwrite('some_eroded_img.jpg', np.asarray(ret_img1, dtype='uint8'))

# Opening and closing to remove noise
img_opening1 = opening(padded_img, structuring_element)
cv2.imwrite('some_opening.jpg', np.asarray(img_opening1, dtype='uint8'))
res_noise1 = closing(img_opening1, structuring_element)
cv2.imwrite('res_noise1.jpg', np.asarray(res_noise1, dtype='uint8'))

# Closing and Opening to remove noise
img_closing2 = closing(padded_img, structuring_element)
cv2.imwrite('some_closing.jpg', np.asarray(img_closing2, dtype='uint8'))
res_noise2 = opening(img_closing2, structuring_element)
cv2.imwrite('res_noise2.jpg', np.asarray(res_noise2, dtype='uint8'))

# Extracting boundaries of res_noise1.jpg
eroded_img1 = erosion(res_noise1, structuring_element_5, size=5)
res_bound1 = np.subtract(res_noise1, eroded_img1)
cv2.imwrite('res_bound1.jpg', np.asarray(res_bound1))

# Extracting boundaries of res_noise2.jpg
eroded_img2 = erosion(res_noise2, structuring_element_5, size=5)
res_bound2 = res_noise2 - eroded_img2
cv2.imwrite('res_bound2.jpg', np.asarray(res_bound2))
