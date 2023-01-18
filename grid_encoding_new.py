import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from grid_encoding import get_all_positions, plot_sim

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

if __name__ == "__main__":
    sum_test = False
    sim_test = False
    plot_embed = True
    hist = False

    # test sum behavior
    encodings = grid_encoding_new(28, 128, factor = 40, random=True)
    if sum_test:
        values = encodings.sum(dim=-1)
        # plt.imshow(encodings[:,:,:3])
        plt.imshow(values)
        plt.show()

    # test plot_sim
    if sim_test:
        plot_sim(encodings[:, :, :])

    if plot_embed:
        plt.imshow(encodings.reshape((28*28, 128)))
        plt.show()
    if hist:
        plt.hist(encodings.reshape((28*28, 128)).flatten(), bins=100)
        plt.show()
