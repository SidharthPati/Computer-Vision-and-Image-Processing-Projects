# __author__ = 'spati'
import cv2
import numpy as np
import matplotlib.pyplot as plt



def calculate_cluster_and_means(red_center, green_center, blue_center, x_matrix):
    """

    :param red_center: Centroid of red cluster
    :param green_center: Centroid of green cluster
    :param blue_center: Centroid of blue cluster
    :param x_matrix: all points
    :return: red_mean, green_mean, blue_mean, red_list, green_list, blue_list
    """
    # Lists containing the points of a cluster
    red_list = []
    green_list = []
    blue_list = []
    red_x = []
    red_y = []
    green_x = []
    green_y = []
    blue_x = []
    blue_y = []

    for coordinate in x_matrix:
        dist_red = np.linalg.norm(np.array(coordinate) - np.array(red_center))
        dist_green = np.linalg.norm(np.array(coordinate) - np.array(green_center))
        dist_blue = np.linalg.norm(np.array(coordinate) - np.array(blue_center))
        min_value = min(dist_red, dist_green, dist_blue)

        if min_value == dist_red:
            red_list.append(coordinate)
            red_x.append(coordinate[0])
            red_y.append(coordinate[1])
        if min_value == dist_green:
            green_list.append(coordinate)
            green_x.append(coordinate[0])
            green_y.append(coordinate[1])
        if min_value == dist_blue:
            blue_list.append(coordinate)
            blue_x.append(coordinate[0])
            blue_y.append(coordinate[1])

    red_mean = [np.mean(np.array(red_x)), np.mean(np.array(red_y))]
    green_mean = [np.mean(np.array(green_x)), np.mean(np.array(green_y))]
    blue_mean = [np.mean(np.array(blue_x)), np.mean(np.array(blue_y))]

    return red_mean, green_mean, blue_mean, red_list, green_list, blue_list


# plot the coordinates
def plot_coordinate(color, point_coordinates, shape=None):
    """
    :param color: color
    :param point_coordinates: coordinates of the point
    :param shape: triangle or circle
    """
    x_coord = []
    y_coord = []
    if shape == "circle":
        label = str(point_coordinates[0]) + "," + str(point_coordinates[1])
        plt.scatter(point_coordinates[0], point_coordinates[1], c=color)
        plt.text(point_coordinates[0], point_coordinates[1], label, family="serif")
    else:
        for pt in point_coordinates:
            x_coord.append(pt[0])
            y_coord.append(pt[1])
            label = str(pt[0]) + "," + str(pt[1])
            plt.text(pt[0], pt[1], label, family="serif")

        # plt.scatter(x_coord, y_coord, marker='^', facecolors="None", edgecolors=color)
        plt.scatter(x_coord, y_coord, marker='^', c=color)


# Matrix X whose rows represent different data points
X_matrix = [[5.9, 3.2],
            [4.6, 2.9],
            [6.2, 2.8],
            [4.7, 3.2],
            [5.5, 4.2],
            [5.0, 3.0],
            [4.9, 3.1],
            [6.7, 3.1],
            [5.1, 3.8],
            [6.0, 3.0]]

# Value of k for k-means clustering
k = 3

# Initial Centers of 3 clusters
u1_red = [6.2, 3.2]
u1_green = [6.6, 3.7]
u1_blue = [6.5, 3.0]

# No. of samples for Task 3.1
N = len(X_matrix)

# Calculate clusters and means
mean_red, mean_green, mean_blue, list_red, list_green, list_blue = calculate_cluster_and_means(
    u1_red, u1_green, u1_blue, X_matrix)

# Plot for 3.1
plt.figure(1)

# Plot the points
print("iter1")
print("Red cluster: ")
print(list_red)
print("Green cluster: ")
print(list_green)
print("Blue cluster: ")
print(list_blue)
plot_coordinate("red", list_red)
plot_coordinate("green", list_green)
plot_coordinate("blue", list_blue)

# Task 3.1 output
plt.savefig('task3_iter1_a.jpg')

# Plot for 3.2
plt.figure(2)

# Plot mean,u coordinates
print("u values")
print("u red:")
print(mean_red)
print("u green:")
print(mean_green)
print("u blue:")
print(mean_blue)
plot_coordinate("red", mean_red, shape="circle")
plot_coordinate("green", mean_green, shape="circle")
plot_coordinate("blue", mean_blue, shape="circle")

# Task 3.2 output
plt.savefig('task3_iter1_b.jpg')

# Plot for 3.3a
plt.figure(3)

# Task 3.3 with new means
mean_red_2, mean_green_2, mean_blue_2, list_red_2, list_green_2, list_blue_2 = calculate_cluster_and_means(
    mean_red, mean_green, mean_blue, X_matrix)

# Plot the points
print("Iter 2")
print("red cluster: ")
print(list_red_2)
print("green cluster: ")
print(list_green_2)
print("blue cluster: ")
print(list_blue_2)
plot_coordinate("red", list_red_2)
plot_coordinate("green", list_green_2)
plot_coordinate("blue", list_blue_2)

# Task 3.3a output
plt.savefig('task3_iter2_a.jpg')

# Plot for 3.3b
plt.figure(4)

# Plot mean,u coordinates
print("mean")
print("red mean")
print(mean_red_2)
print("green mean")
print(mean_green_2)
print("red mean")
print(mean_blue_2)
plot_coordinate("red", mean_red_2, shape="circle")
plot_coordinate("green", mean_green_2, shape="circle")
plot_coordinate("blue", mean_blue_2, shape="circle")

# Task 3.3b output
plt.savefig('task3_iter2_b.jpg')
