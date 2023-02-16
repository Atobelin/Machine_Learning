"""
Machine Learning Assignment 1

@author Yanlin Mi R1202208
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def read_data():
    """
    Read data from a csv file, return a numpy matrix.
    In my workspace, my csv file is under the same path as the py file.
    """
    data = np.genfromtxt("./clusteringData.csv", delimiter=",")
    return data


def create_centroids(feature_data, k):
    """
    Create k initial centroids.
    Use np.random.choice to choose k random rows.
    Return a 8*k matrix. Here features number = 8.
    """
    row_total = feature_data.shape[0]
    row_sequence = np.random.choice(row_total, k, replace=False, p=None)
    return feature_data[row_sequence, :]


def calculate_distance(feature_data, query_point):
    """
    Calculate distances from query_point to each point in feature_data.
    Use np.apply_along_axis to apply a function to each row.
    There is a lambda function for shorter and more clear code.
    Use np.abs(p1 - p2).sum() to calculate Manhattan distance.
    Return a 1d array, the length equals to feature_data's length.
    """
    distances = np.apply_along_axis(
        lambda p: np.abs(p - query_point).sum(), 1, feature_data
    )
    return distances


def assign_centroids(feature_data, centroids):
    """
    Assign centroid to each point in the feature_data.
    Calculate distances to each centroid and pick the closest.
    Return a 1d array and the item in the array is centroids index.
    """
    distances = []
    for centroid in centroids:
        distances.append(calculate_distance(feature_data, centroid))
    return np.argmin(distances, axis=0)


def move_centroids(feature_data, min_distance, centroids):
    """
    Move centroids to new centroids.
    Choose all the points belong to this centroid and calculate the mean.
    Return k new centroids.
    """
    new_centroids = []
    for i in range(len(centroids)):
        points = feature_data[min_distance == i, :]
        new_centroid = np.mean(points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


def distortion_cost(feature_data, min_distance, centroids):
    """
    Calculate distances between a centroid and each points belong to this centroid.
    Then calculate the sum of the squares of these distances.
    Return an integer.
    """
    cost = 0
    for i in range(len(centroids)):
        points = feature_data[min_distance == i, :]
        distance = calculate_distance(points, centroids[i])
        cost += (distance**2).sum()
    return cost


def restart_KMeans(feature_data, centroids_num, iterations_num, restart_num):
    """
    There are two iterations, the inner one contains move_centroids and assign_centroids.
    The outer one contains the controids initialisation, the inner iteration and the cost calculation.
    Then choose a best cost and return.
    """
    best_cost = math.inf
    for i in range(restart_num):
        print("restart", i + 1, "times", "k =", centroids_num)
        centroids = create_centroids(feature_data, centroids_num)
        min_distance = assign_centroids(feature_data, centroids)
        for _ in range(iterations_num):
            centroids = move_centroids(feature_data, min_distance, centroids)
            min_distance = assign_centroids(feature_data, centroids)
        cost = distortion_cost(feature_data, min_distance, centroids)
        if best_cost > cost:
            best_cost = cost
            best_centroids = centroids
    return (best_centroids, best_cost)


def main():
    """
    The main function.
    Read data.
    Run restart_KMeans function ten times.
    Draw a elbow plot.
    """
    feature_data = read_data()
    k = range(1, 10)
    distortions = []
    for k_number in k:
        print("start calculate, k =", k_number)
        best_centroids, best_cost = restart_KMeans(
            feature_data, k_number, 10, 10)
        print("k=", k_number, best_centroids, best_cost)
        distortions.append(best_cost)
    plt.plot(k, distortions)
    plt.xlabel("Number of K")
    plt.ylabel("Distortion")
    plt.savefig("./plot.jpg")
    plt.show()


main()
