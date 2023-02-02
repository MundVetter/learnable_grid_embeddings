import torch as tc
import torch
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize

def special_dot(t, k):
    return t[..., 0] * k[0] + t[..., 1] * k[1]


def get_all_positions(n):
    x = tc.arange(0.0, n)
    y = tc.arange(0.0, n)
    return tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

def get_all_positions_3d(n):
    x = tc.arange(0.0, n)
    y = tc.arange(0.0, n)
    z = tc.arange(0.0, n)
    return tc.stack(tc.meshgrid(x, y, z), dim=-1).reshape(-1, 3)

def calculate_div_term(d_model, factor):
    return tc.exp(tc.arange(0, d_model).float() * (-math.log(factor) / d_model))

def hexagon_encoding_new(r, r0, lamda, theta):
    freq = 2/(math.sqrt(3)*lamda)
    X = r[..., 0]
    Y = r[..., 1]
    x0 = r0[..., 0]
    y0 = r0[..., 1]
    orien = theta

    G = 1/3 * ( 
    2/3 * (
        torch.cos(2*torch.pi*freq * ((X-x0)*torch.cos(2/3*np.pi*1+orien) + (Y-y0)*torch.sin(2/3*torch.pi*1+orien)))
        + torch.cos(2*torch.pi*freq * ((X-x0)*torch.cos(2/3*np.pi*2+orien) + (Y-y0)*torch.sin(2/3*torch.pi*2+orien)))
        + torch.cos(2*torch.pi*freq * ((X-x0)*torch.cos(2/3*np.pi*3+orien) + (Y-y0)*torch.sin(2/3*torch.pi*3+orien)))
            )
    + 1
        )

    return G

def hexagon_encoding(t, func=tc.sin, offset=0):
    # create three 2 dimensional unit vectors that are 120 degrees apart
    # k1 = tc.tensor([math.sqrt(2)/ 2, math.sqrt(2)/ 2])
    # k2 = tc.tensor([-(math.sqrt(6) + math.sqrt(2))/4, (math.sqrt(6) - math.sqrt(2))/4])
    # k3 = tc.tensor([(math.sqrt(6) - math.sqrt(2))/4, - (math.sqrt(6) + math.sqrt(2))/4])
    k1 = tc.tensor([1.0, 0.0])
    k2 = tc.tensor([-1/2, 3**0.5/2])
    k3 = tc.tensor([-1/2, -3**0.5/2])

    t += offset

    encoding = func(special_dot(t, k1)) + func(special_dot(t, k2)) + func(special_dot(t, k3))

    return encoding

def hexagon_1_encoding(t, func=tc.sin):
    return hexagon_encoding(t, func=func, offset=1)

def hexagon_n14_encoding(t, func=tc.sin):
    return hexagon_encoding(t, func=func, offset=-14)

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

def rotate_by_degrees(t, degrees):
    """ Rotate a tensor t (nxfx2) by a given number of degrees (fx1). """
    radians = degrees * math.pi / 180
    c, s = tc.cos(radians), tc.sin(radians)
    R = tc.stack((c, -s, s, c), dim=1).view(-1, 2, 2)
    R = R.repeat(t.shape[0], 1, 1, 1)
    return (R @ t.type(R.type()).unsqueeze(3)).squeeze(3)

def generate_positional_encoding(max_len, dim, factor=10_000, encode_function=hexagon_encoding, rotation=0, offset=0, random=False, cosine=False, scale=1.0):
    """ Generate position encoding for a given max_len and d_model.
    """
    temperature = factor

    position = get_all_positions(max_len)
    # div_term = calculate_div_term(dim, factor)
    if cosine:
        dim = dim // 2

    omega = torch.arange(dim) / (dim - 1)
    omega = 1. / (temperature ** omega)
    # change div term such that [x,y,..., z] -> [[x,x], [y,y], ..., [z,z]]
    omega = tc.stack([omega, omega], dim=-1)

    # multiply by div term to get (d_model, 100, 2) tensor
    position_scales = position.unsqueeze(1) * omega * scale

    if random:
        rotations = tc.rand(dim, generator=torch.Generator().manual_seed(2147483647)) * 360
    else:
        rotations = torch.unsqueeze(tc.arange(0, dim, 1) * rotation, dim=1)
    # generate random rotations in degrees
    position_scales = rotate_by_degrees(position_scales, rotations)

    encodings_sin = encode_function(position_scales)
    if cosine:
        encodings_cos = encode_function(position_scales, func=tc.cos)
        # merge sin and cos encodings
        encodings = tc.stack([encodings_sin, encodings_cos], dim=-1)
        dim *= 2
    else:
        encodings = encodings_sin

    return encodings.reshape(max_len, max_len, dim)

def generate_position_encoding_old(max_len, dim, factor=100, encode_function=hexagon_encoding):
    """ Generate position encoding for a given max_len and d_model.
    """
    position = get_all_positions(max_len)
    div_term = calculate_div_term(dim, factor)
    # change div term such that [x,y,..., z] -> [[x,x], [y,y], ..., [z,z]]
    div_term = tc.stack([div_term, div_term], dim=-1)

    # multiply by div term to get (d_model, 100, 2) tensor
    position_scales = position.unsqueeze(1) * div_term

    encodings = encode_function(position_scales, offset=offset)

    return encodings.reshape(max_len, max_len, dim)

def dot_product_sim(encoding_grid, position):
    cell = encoding_grid[position[0], position[1], :]

    # calculate dot product between center and all other points
    dot_product = tc.einsum('d, hwd -> hw', cell, encoding_grid)
    # normalize
    dot_product = dot_product / tc.norm(cell) / tc.norm(encoding_grid, dim=-1)
    return dot_product

import torch

# def gini(values):
#     sorted_values = torch.sort(values)[0]
#     height, area = 0, 0
#     for value in sorted_values:
#         height += value
#         area += height - value / 2.
#     fair_area = height * values.size(0) / 2.
#     return (fair_area - area) / fair_area


def loss_fun(args):
    rotation, offset = args
    encodings = generate_positional_encoding(28, 128, factor = 10_000, encode_function=hexagon_encoding, rotation=rotation)
    values = encodings.sum(dim=-1).reshape(-1)
    return (values * values).std()

def plot_sim(encodings =generate_positional_encoding(28, 512, 10000, encode_function=hexagon_encoding, offset = 1, random=True, cosine=True), pos= [[14, 14], [5, 5], [5, 23], [23, 5], [23, 23]]):
    for p in pos:
        sim = dot_product_sim(encodings, p)
        plt.imshow(sim)
        plt.show()



if __name__ == "__main__":
    # sum_test = False
    # sim_test = True
    # plot_embed = False

    # # test sum behavior
    # encodings = generate_positional_encoding(28, 128, factor = 10_000, encode_function=hexagon_n14_encoding, rotation=0, random=False, cosine=True)
    # if sum_test:
    #     values = encodings.sum(dim=-1)
    #     # plt.imshow(encodings[:,:,:3])
    #     plt.imshow(values)
    #     plt.show()

    # # test plot_sim
    # if sim_test:
    #     plot_sim(encodings[:, :, :])

    # if plot_embed:
    #     plt.imshow(encodings.reshape((28*28, 128)))
    #     plt.show()


    # k1 = tc.tensor([math.sqrt(2)/ 2, math.sqrt(2)/ 2])
    # k2 = tc.tensor([-(math.sqrt(6) + math.sqrt(2))/4, (math.sqrt(6) - math.sqrt(2))/4])
    # k3 = tc.tensor([(math.sqrt(6) - math.sqrt(2))/4, - (math.sqrt(6) + math.sqrt(2))/4])

    # # plot the vectors
    # plt.plot([0, k1[0]], [0, k1[1]], 'r')
    # plt.plot([0, k2[0]], [0, k2[1]], 'r')
    # plt.show()

    # print(triangle_encoding(tc.tensor([[0, 500]])))

    # # # plot the encoding
    # encodings = generate_positional_encoding(28, 512, factor = 10_000, encode_function=hexagon_encoding, rotation=60, offset=0)
    # values = encodings.sum(dim=-1)
    # plt.imshow(encodings[:,:,:3])
    # # plt.imshow(values)
    # plt.show()

    # # print(gini(values))
    # offset_range = 28
    # angle_range = 180
    # data = []
    # step_offset = 1
    # step_angle = 0.1
    # for j in np.arange(-offset_range, offset_range, step_offset):
    #     results = []
    #     for i in np.arange(0.0, angle_range, step_angle):
    #         encodings = generate_positional_encoding(28, 256, factor = 10_000, encode_function=hexagon_encoding, rotation=torch.tensor(i).double(), offset=j)
    #         values = encodings.sum(dim=-1).reshape(-1)
    #         results.append((values * values).std())
    #     # print lowest
    #     print(np.argmin(results), j, np.min(results))
    #     data.append(results)
    
    # # # # # # get x,y of lowest value
    # print("LOWEST VALUE")
    # data = np.array(data)
    # print(np.unravel_index(np.argmin(data), data.shape))
    # print(np.min(data))

    # # # # # # MAKE 3dPLOT OF RESULTS
    # x = np.arange(0.0, angle_range, step_angle)
    # y = np.arange(-offset_range, offset_range, step_offset)
    # X, Y = np.meshgrid(x, y)
    # Z = data

    # # save data
    # np.save("data.npy", data)

    # results = []
    # for i in np.arange(0.0, angle_range, step_angle):
    #     encodings = generate_positional_encoding(28, 512, factor = 10000, encode_function=hexagon_encoding, rotation=torch.tensor(i).double(), offset=1)
    #     values = encodings.sum(dim=-1).reshape(-1)
    #     results.append((values * values).std())

    # print(np.argmin(results), np.min(results))

    # # # plot
    # plt.plot(np.arange(0.0, angle_range, step_angle), results)
    # plt.show()


    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), cmap='viridis', edgecolor='none')
    # plt.title('Variance of positional encoding')
    # plt.show()


    # x = generate_positional_encoding(28, 128, factor = 10_000, encode_function=hexagon_1_encoding, rotation=357.3)
    # plt.imshow(x.sum(dim=-1))
    # plt.show()
    # print(optimize.minimize(loss_fun, [15, 15], method='Nelder-Mead', options={"maxiter":5000}))

    # print(loss_fun(torch.tensor(11).double()))
    # plt.imshow(encodings.sum(dim=-1))
    # plt.show()

    # encodings_old = generate_positional_encoding(100, 4, 10_000,encode_function=hexagon_encoding, offset = 0, rotation=0)
    # plt.imshow(encodings_old)
    # plt.show()

    x = tc.arange(0, 40, 1)
    y = tc.arange(0, 40, 1)

    t = tc.stack(tc.meshgrid(x, y), dim=-1).reshape(-1, 2)

    # theta = tc.tensor([2])
    # lamda = 10
    # r0 = tc.tensor([10, 10])
    # z = hexagon_encoding_new(t, r0, lamda, theta)


    z =hexagon_encoding(t)
    # z = square_encoding(t)
    # # z = triangle_encoding(t)
    # # print(z)

    ax = plt.axes(projection='3d')
    ax.view_init(azim=0, elev=90)
    ax.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    plt.title('Hexagon Encoding')
    plt.show()


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
