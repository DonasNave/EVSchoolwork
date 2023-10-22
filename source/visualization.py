import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_function(func, bounds=(-100, 100), num_points=140):
    vec_x = np.linspace(bounds[0], bounds[1], num_points)
    vec_y = np.linspace(bounds[0], bounds[1], num_points)

    # Create a figure with separate subplots for 2D and 3D
    fig = plt.figure(figsize=(12, 5))

    # 2D plot
    ax1 = fig.add_subplot(121)
    y = [func([xi]) for xi in vec_x]
    ax1.plot(vec_x, y)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Function Value")
    ax1.set_title("2D Function Visualization")

    # 3D plot
    z = np.zeros_like(vec_x)

    X, Y = np.meshgrid(vec_x, vec_y)
    Z = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = func([X[i, j], Y[i, j]])

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, Z)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Function Value", labelpad=-25)
    ax2.set_title("3D Function Visualization")

    # Set the suptitle with custom name and source
    custom_name = getattr(func, "_custom_name", "Function")
    fig.suptitle(custom_name)
    plt.tight_layout()
    plt.show()
