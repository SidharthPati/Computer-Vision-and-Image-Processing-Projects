# __author__ = 'spati'
import cv2
import numpy as np


# Function to calculate R, G, B distance
def calc_distance(rgb_list1, rgb_list2):
    """

    :param rgb_list1: 1st list containing R, G, B values
    :param rgb_list2: 2nd list containing R, G, B values
    :return: distance between the above two R, G, B values
    """
    dist = (((rgb_list1[0] - rgb_list2[0]) ** 2) +
            ((rgb_list1[1] - rgb_list2[1]) ** 2) +
            ((rgb_list1[2] - rgb_list2[2]) ** 2)) ** 0.5
    return dist


# Function to calculate means of R, G, B
def calc_mean(list_group, r_mat, g_mat, b_mat):
    """

    :param list_group: list of all clusters
    :param r_mat: matrix of red pixel values
    :param g_mat: matrix of green pixel values
    :param b_mat: matrix of blue pixel values
    :return: means of r,g,b values
    """
    if len(list_group) == 0:
        return 9999, 9999, 9999
    sum_r_val, sum_b_val, sum_g_val = 0, 0, 0

    for i in range(len(list_group)):
        x = list_group[i][0]
        y = list_group[i][1]
        sum_r_val+= r_mat[x][y]
        sum_b_val+= b_mat[x][y]
        sum_g_val+= g_mat[x][y]

    mean_r_val = sum_r_val//len(list_group)
    mean_g_val = sum_g_val//len(list_group)
    mean_b_val = sum_b_val//len(list_group)

    return mean_r_val,  mean_g_val, mean_b_val


k_values = [3, 5, 10, 20]

# reading Baboon image
baboon_img = cv2.imread("baboon.jpg")

# read R, G, B values from the image and store them in separate lists
r_matrix = baboon_img[:, :, 2]
g_matrix = baboon_img[:, :, 1]
b_matrix = baboon_img[:, :, 0]

image_ht, image_wd, t = baboon_img.shape

for k in k_values:
    # baboon_new = [[0 for j in range(image_wd)] for i in range(image_ht)]

    # copying original image to new baboon image
    baboon_new = np.copy(np.asarray(baboon_img))

    # list of previous mean color values
    old_rgb_mean = [0 for i in range(k)]

    # Flag becomes True when the previous mean list and the present mean list are equal
    mean_check = False

    # pick random values for the 1st iteration
    first_iteration = 1

    while mean_check is False:

        # empty list for storing new mean
        latest_rgb_mean = [0 for i in range(k)]

        # iterate for all k values
        for k1 in range(k):
            if first_iteration == 1:
                # take random values
                latest_rgb_mean[k1] = np.random.randint(low=0, high=255, size=1), \
                                      np.random.randint(low=0, high=255, size=1),\
                                      np.random.randint(low=0, high=255, size=1)
            else:
                latest_rgb_mean[k1] = calc_mean(k_clusters[k1], r_matrix, g_matrix, b_matrix)

        # if new and old iteration's means are same, then stop
        if old_rgb_mean == latest_rgb_mean:
            mean_check = True
            break

        # save current values in old_rgb_mean to use it in next iteration
        old_rgb_mean = latest_rgb_mean
        first_iteration += 1
        k_clusters = [[] for i in range(k)]

        for y in range(image_ht):
            for x in range(image_wd):
                k1 = 0
                dist_list = [0 for i in range(k)]

                for k1 in range(k):
                    point_rgb = r_matrix[x][y], g_matrix[x][y], b_matrix[x][y]
                    u_rgb = latest_rgb_mean[k1]

                    dist_list[k1] = calc_distance(point_rgb, u_rgb)
                dist_np = np.asarray(dist_list)
                minima = dist_list.index(min(dist_list))
                k_clusters[minima].append((x, y))

    for index in range(k):
        # print(len(clusters_array[k_index]))
        if latest_rgb_mean[index][0] != 9999:
            rgb = r_matrix[latest_rgb_mean[index][0]][latest_rgb_mean[index][1]], \
                  g_matrix[latest_rgb_mean[index][0]][latest_rgb_mean[index][1]], \
                  b_matrix[latest_rgb_mean[index][0]][latest_rgb_mean[index][1]]
            for c_iterator in range(len(k_clusters[index])):
                x = k_clusters[index][c_iterator][0]
                y = k_clusters[index][c_iterator][1]
                baboon_new[x][y] = rgb

        baboon_new = np.asarray(baboon_new, dtype="float32")
        cv2.imwrite("task3_baboon_" + str(k) + ".jpg", baboon_new)
