import torch as tc
import numpy as np
from matplotlib import pyplot as plt

def simple_dot(t, k):
    return t[:, 0] * k[0] + t[:, 1] * k[1]

def grid_encoding(t):
    # create three 2 dimensional unit vectors that are 120 degrees apart
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([-1/2, 3**0.5/2])
    k3 = tc.tensor([-1/2, -3**0.5/2])

    encoding = tc.sin(simple_dot(t, k1)) + tc.sin(simple_dot(t, k2)) + tc.sin(simple_dot(t, k3))

    return encoding

def square_encoding(t):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.0, 1.0])

    encoding = tc.sin(simple_dot(t, k1)) + tc.sin(simple_dot(t, k2))

    return encoding

def triangle_encoding(t):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.5, 3**0.5/2])

    encoding = tc.sin(simple_dot(t, k1)) + tc.sin(simple_dot(t, k2))

    return encoding

if __name__ == "__main__":
    # k1 = tc.tensor([1.0, 0.0])
    # k2 = tc.tensor([0.5, 3**0.5/2])
    # # angle between k1 and k2 in degrees
    # angle = np.arccos(tc.dot(k1, k2) / (tc.norm(k1) * tc.norm(k2))) * 180 / np.pi
    # print(angle)

    # plot the vectors
    plt.plot([0, k1[0]], [0, k1[1]], 'r')
    plt.plot([0, k2[0]], [0, k2[1]], 'r')
    plt.show()

    x = tc.arange(0, 40, 1)
    y = tc.arange(0, 40, 1)

    t = tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

    # z = grid_encoding(t)
    # z = square_encoding(t)
    z = triangle_encoding(t)
    print(z)

    ax = plt.axes(projection='3d')
    ax.view_init(azim=0, elev=90)
    ax.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    plt.show()
