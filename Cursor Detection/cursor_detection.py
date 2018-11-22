# __author__ = 'spati'
# References:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
# https://pythonprogramming.net/template-matching-python-opencv-tutorial/
# https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
# https://docs.opencv.org/3.4.2/de/da9/tutorial_template_matching.html

import cv2 as cv
import glob


def image_show(image1, window_name= 'image'):
    """
    Shows the image on rectangular window
    :param image1: Image to be shown
    :param window_name: Name of the window
    :return: None
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow('image', image1)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Read the positive images
positive_images = [cv.imread(file, cv.IMREAD_COLOR) for file in glob.glob("task3/pos/*.jpg")]

# Create window
cv.namedWindow('image', cv.WINDOW_NORMAL)

# Read the template
template = cv.imread("task3/template_original.png", 0)
cv.namedWindow('template', cv.WINDOW_NORMAL)

# Resize the template
template = cv.resize(template, (0, 0), fx=0.58, fy=0.58)
template_row, template_column = template.shape[::-1]

# Calculate Laplacian of Template
laplacian_template = cv.Laplacian(template, cv.CV_32F)

# Set Threshold of template
ret2, threshold_template = cv.threshold(laplacian_template, 25, 255, cv.THRESH_BINARY)

i = 0
for image in positive_images:
    # Blur the image using GaussianBlur
    blurred_image = cv.GaussianBlur(image, (3, 3), 0)

    # Convert to Grayscale image
    gray_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

    # Calculate Laplacian of Image
    laplacian_image = cv.Laplacian(gray_image, cv.CV_32F)

    # Set threshold of image
    ret1, threshold_image = cv.threshold(laplacian_image, 25, 255, cv.THRESH_BINARY)

    # Template matching using TM_CCOEFF_NORMED
    res = cv.matchTemplate(threshold_image, threshold_template, cv.TM_CCOEFF_NORMED)

    # Calculate min and max locations to create the rectangle of the image
    min_value, max_value, min_location, max_location = cv.minMaxLoc(res)
    top_left = max_location
    bottom_right = (top_left[0] + template_row, top_left[1] + template_column)
    cv.rectangle(image,top_left, bottom_right, 255, 2)
    cv.namedWindow('Final Result', cv.WINDOW_NORMAL)
    cv.imshow('Final Result', image)
    filename = "DetectedCursors/cursor"+str(i) +".jpg"
    cv.imwrite(filename, image)
    cv.waitKey(0)
    i = i+1
    cv.destroyWindow('Final Result')
