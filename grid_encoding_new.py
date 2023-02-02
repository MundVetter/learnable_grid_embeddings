import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from grid_encoding import get_all_positions, plot_sim, get_all_positions_3d
from mayavi import mlab

def hexagon_encoding_new(r, r0, omega, theta):
    freq = 2/(math.sqrt(3)*omega)
    # freq = omega
    X = r[..., 0].unsqueeze(-1)
    Y = r[..., 1].unsqueeze(-1)
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

def grid_encoding_new(max_len, dim, factor = 10_000, seed = 2147483647, random = False, lamda_lower = 0.28, lamda_upper = 1.0, spacing_ratio = 1.42):
    position = get_all_positions(max_len)


    if random:
        c = math.log(lamda_upper/lamda_lower) / math.log(spacing_ratio)
        i = torch.linspace(0.0, c, dim)
        omega = factor * 0.28 * 1.42 ** i
        generator = torch.Generator().manual_seed(seed)

        thetas = np.pi / 3 * torch.rand(dim, generator=generator)
        r0s = torch.rand(dim, 2, generator=generator) * torch.stack([omega, omega], dim=-1)
    else:
        num_lambda = int(np.log(1/0.28) / np.log(spacing_ratio)) + 1
        i = torch.arange(num_lambda)
        num_thetas = 8
        thetas = torch.arange(0, np.pi / 3, np.pi / 3 / num_thetas)
        num_r0s = 2
        x = torch.arange(0.0, 1.0, 1 / num_r0s)
        y = torch.arange(0.0, 1.0, 1 / num_r0s)
        omega = factor * 0.28 * 1.42 ** i
        parms = torch.cartesian_prod(omega, thetas, x, y)
        omega = parms[:, 0]
        thetas = parms[:, 1]
        r0s = parms[:, 2:] * torch.stack([omega, omega], dim=-1)

    G = hexagon_encoding_new(position, r0s, omega, thetas)

    return G.reshape(max_len, max_len, dim)

def input_encoder(x, a, b):
    return torch.cat([a * torch.sin((2.*np.pi*x.type(b.type())) @ b.T), a * torch.cos((2.*np.pi*x.type(b.type())) @ b.T)], dim=-1)

input_encoder_3d = lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x) @ b.T), 
                                                a * np.cos((2.*np.pi*x) @ b.T)], axis=-1) 

def rotate(v, rad):
    return torch.stack([math.cos(rad)*v[:, 0] - math.sin(rad)*v[:, 1], math.sin(rad)*v[:, 0] + math.cos(rad)*v[:, 1]], dim=-1)

def grid_encoder(x, a, b):
    b1 = b
    b2 = rotate(b, 2/3*np.pi)
    b3 = rotate(b, 4/3*np.pi)

    return input_encoder(x, a, b1) + input_encoder(x, a, b2) + input_encoder(x, a, b3)

def rotate_x(v, theta):
    # theta: rotation angle around x axis
    # v: [N, 3]
    # return: [N, 3]
    return v @ np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def rotate_y(v, theta):
    # theta: rotation    around y axis
    # v: [N, 3]
    # return: [N, 3]
    return v @ np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rotate_z(v, theta):
    # theta: rotation angle around z axis
    # v: [N, 3]
    # return: [N, 3]
    return v @ np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def generate_rotation_matrix(p0, p1):
    C = np.cross(p0, p1)
    D = np.dot(p0, p1)
    NP0 = np.linalg.norm(p0)

    if not all(C == 0):
        Z = np.array([[0, -C[2], C[1]], [C[2], 0, -C[0]], [-C[1], C[0], 0]])
        R = (np.eye(3) + Z + Z @ Z * (1 - D) / np.linalg.norm(C) ** 2) / NP0 ** 2
    else:
        R = np.sign(D) * (np.linalg.norm(p1) / NP0)
    return R

def grid_encoding_3d(x, a, b):
    # b1 = b
    # b2 = rotate_y(b1, -np.arccos(-1/3))
    # b3 = rotate_z(b2, -2*np.pi/3)
    # b4 = rotate_z(b2, 2*np.pi/3)

    # return (input_encoder(x, a, b1) + input_encoder(x, a,b2) + input_encoder(x, a, b3) + input_encoder(x, a, b4))/4
    

    # calculate the norm of b 
    norm_b = np.linalg.norm(b)
    scaling_factor = norm_b / np.sqrt(3/2)

    b1 = np.array([0, 0, np.sqrt(3/2)])
    b2 = np.array([2/np.sqrt(3), 0, -1/np.sqrt(6)])
    b3 = np.array([-1/np.sqrt(3), 1, -1/np.sqrt(6)])
    b4 = np.array([-1/np.sqrt(3), -1, -1/np.sqrt(6)])

    # calculate the rotations matrix
    R = generate_rotation_matrix(b1, b)
    b2_fixed = R @ b2 * scaling_factor
    b3_fixed = R @ b3 * scaling_factor
    b4_fixed = R @ b4 * scaling_factor

    return (input_encoder_3d(x, a, b) + input_encoder_3d(x, a, b2_fixed) + input_encoder_3d(x, a, b3_fixed) + input_encoder_3d(x, a, b4_fixed))/4


def generate_encoding(max_len, a, b, fun = grid_encoder):
    position = get_all_positions(max_len) / max_len
    return fun(position, a, b).reshape(max_len, max_len, -1)

def generate_encoding_3d(max_len, a, b, fun = grid_encoding_3d):
    position = get_all_positions_3d(max_len) / max_len
    return fun(position, a, b).reshape(max_len, max_len, max_len, -1)


if __name__ == "__main__":
    sum_test = False
    sim_test = False
    plot_embed = False
    hist = False
    d3_plot = False
    d3_plot_2d = True

    max_len = 40
    # test sum behavior
    # encodings = grid_encoding_new(28, 128, factor = 40, random=True)
    encodings = generate_encoding_3d(max_len, 1.0, torch.tensor([0, np.sqrt(3/2), 0])* 1)
    if sum_test:
        values = encodings.sum(dim=-1)
        # plt.imshow(encodings[:,:,:3])
        plt.imshow(values)
        plt.show()

    # test plot_sim
    if sim_test:
        plot_sim(encodings[:, :, :])

    if plot_embed:
        plt.imshow(encodings.reshape((100*100, 128)))
        plt.show()
    if hist:
        plt.hist(encodings.reshape((28*28, 128)).flatten(), bins=100)
        plt.show()
    if d3_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot all values where [;,;,;,0] is above 0.8
        grid_cells = encodings[:,:,:,0] > 0.8
        for x in range(max_len):
            for y in range(max_len):
                for z in range(max_len):
                    if grid_cells[x,y,z]:
                        ax.scatter(x, y, z)

    if d3_plot_2d:
        # show the activations in 2d by looping over the z axis
        x = torch.arange(0, max_len, 1)
        y = torch.arange(0, max_len, 1)

        t = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2)
        for z in range(max_len):
            ax = plt.axes(projection='3d')
            ax.view_init(azim=0, elev=90)
            ax.plot_trisurf(t[:, 0], t[:, 1], encodings[z, :, :, 0].reshape(max_len*max_len), cmap='viridis', edgecolor='none')
            plt.title('Hexagon Encoding sin')
            plt.show()
        # ax.scatter()
