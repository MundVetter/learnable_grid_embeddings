import torch as tc
import torch
import numpy as np
from matplotlib import pyplot as plt
import math

def special_dot(t, k):
    return t[..., 0] * k[0] + t[..., 1] * k[1]


def get_all_positions(n):
    x = tc.arange(0.0, n)
    y = tc.arange(0.0, n)
    return tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

def calculate_div_term(d_model, factor):
    return tc.exp(tc.arange(0, d_model).float() * (-math.log(factor) / d_model))

def hexagon_encoding(t, func=tc.sin):
    # create three 2 dimensional unit vectors that are 120 degrees apart
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([-1/2, 3**0.5/2])
    k3 = tc.tensor([-1/2, -3**0.5/2])

    t += 1

    encoding = func(special_dot(t, k1)) + func(special_dot(t, k2)) + func(special_dot(t, k3))

    return encoding

def square_encoding(t, func=tc.sin):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.0, 1.0])

    encoding = func(special_dot(t, k1)) + func(special_dot(t, k2))

    return encoding

def triangle_encoding(t, func=tc.sin):
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([0.5, 3**0.5/2])

    encoding = func(special_dot(t, k1)) + func(special_dot(t, k2))

    return encoding

def generate_positional_encoding(max_len, dim, factor=10_000, encode_function=hexagon_encoding):
    """ Generate position encoding for a given max_len and d_model.
    """
    temperature = factor
    # y, x = torch.meshgrid(torch.arange(max_len), torch.arange(max_len), indexing = 'ij')
    position = get_all_positions(max_len)
    # div_term = calculate_div_term(dim, factor)
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)
    # change div term such that [x,y,..., z] -> [[x,x], [y,y], ..., [z,z]]
    omega = tc.stack([omega, omega], dim=-1)

    # multiply by div term to get (d_model, 100, 2) tensor
    # y = y.flatten()[:, None] * omega[None, :]
    # x = x.flatten()[:, None] * omega[None, :] 
    position_scales = position.unsqueeze(1) * omega

    encodings_sin = encode_function(position_scales)
    encodings_cos = encode_function(position_scales, func=tc.cos)

    # merge sin and cos encodings
    encodings = tc.stack([encodings_sin, encodings_cos], dim=-1).reshape(max_len, max_len, dim)

    return encodings

def generate_position_encoding_old(max_len, dim, factor=100, encode_function=hexagon_encoding):
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
    encodings = generate_positional_encoding(100, 16, encode_function=hexagon_encoding)
    plt.imshow(encodings[:, :, :3])
    plt.show()
    # encodings_old = generate_position_encoding_old(100, 4, 10_000,encode_function=hexagon_encoding)
    # plt.imshow(encodings_old)
    # plt.show()

    # x = tc.arange(0, 40, 1)
    # y = tc.arange(0, 40, 1)

    # t = tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

    # z =hexagon_encoding(t)
    # # # z = square_encoding(t)
    # # z = triangle_encoding(t)
    # # print(z)

    # ax = plt.axes(projection='3d')
    # ax.view_init(azim=0, elev=90)
    # ax.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    # plt.title('Hexagon Encoding sin')
    # plt.show()


    # z =hexagon_encoding(t, func=tc.cos)
    # # z = square_encoding(t)
    # z = triangle_encoding(t)
    # print(z)

    # ax = plt.axes(projection='3d')
    # ax.view_init(azim=0, elev=90)
    # ax.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    # plt.title('Hexagon Encoding sin')
    # plt.show()

    # # plot old
    # ax2 = plt.axes(projection='3d')
    # ax2.view_init(azim=0, elev=90)
    # ax2.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    # plt.title('Hexagon Encoding old')
    # plt.show()