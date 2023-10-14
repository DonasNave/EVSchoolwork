import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_function(func, bounds=(-100, 100), num_points=100):
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = [func([xi]) for xi in x]

    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Function Value")
    plt.title("2D Function Visualization")
    plt.grid(True)
    plt.show()

def visualize_3d_function(func, bounds=(-100, 100), num_points=100):
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = np.linspace(bounds[0], bounds[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = func([X[i, j], Y[i, j]])  # Replace with your own function

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Function Value")
    ax.set_title("3D Function Visualization")
    plt.show()
