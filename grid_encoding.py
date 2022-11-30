import torch as tc
import numpy as np
from matplotlib import pyplot as plt
import math

def special_dot(t, k):
    return t[..., 0] * k[0] + t[..., 1] * k[1]


def calculate_div_term(d_model, factor):
    return tc.exp(tc.arange(0, d_model).float() * (-math.log(factor) / d_model))

def get_all_positions(n):
    x = tc.arange(0.0, n)
    y = tc.arange(0.0, n)
    return tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

def hexagon_encoding(t):
    # create three 2 dimensional unit vectors that are 120 degrees apart
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([-1/2, 3**0.5/2])
    k3 = tc.tensor([-1/2, -3**0.5/2])

    encoding = tc.sin(special_dot(t, k1)) + tc.sin(special_dot(t, k2)) + tc.sin(special_dot(t, k3))

    return encoding

def square_encoding(t):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.0, 1.0])

    encoding = tc.sin(special_dot(t, k1)) + tc.sin(special_dot(t, k2))

    return encoding

def triangle_encoding(t):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.5, 3**0.5/2])

    encoding = tc.sin(special_dot(t, k1)) + tc.sin(special_dot(t, k2))

    return encoding

def generate_positional_encoding(max_len, dim, factor=100, encode_function=hexagon_encoding):
    """ Generate position encoding for a given max_len and d_model.
    """
    position = get_all_positions(max_len)
    div_term = calculate_div_term(dim, factor)
    # change div term such that [x,y,..., z] -> [[x,x], [y,y], ..., [z,z]]
    div_term = tc.stack([div_term, div_term], dim=-1)

    # multiply by div term to get (d_model, 100, 2) tensor
    position_scales = position.unsqueeze(1) * div_term

    encodings = encode_function(position_scales)

    return encodings.reshape(max_len, max_len, dim)


if __name__ == "__main__":
    # k1 = tc.tensor([1.0, 0.0])
    # k2 = tc.tensor([0.5, 3**0.5/2])

    # # plot the vectors
    # plt.plot([0, k1[0]], [0, k1[1]], 'r')
    # plt.plot([0, k2[0]], [0, k2[1]], 'r')
    # plt.show()

    # print(triangle_encoding(tc.tensor([[0, 500]])))

    # # plot the encoding
    # encodings = generate_position_encoding(100, 3, 10,encode_function=hexagon_encoding)
    # plt.imshow(encodings)
    # plt.show()

    x = tc.arange(0, 40, 1)
    y = tc.arange(0, 40, 1)

    t = tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

    z =hexagon_encoding(t)
    # # z = square_encoding(t)
    # z = triangle_encoding(t)
    # print(z)

    ax = plt.axes(projection='3d')
    ax.view_init(azim=0, elev=90)
    ax.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    plt.show()
