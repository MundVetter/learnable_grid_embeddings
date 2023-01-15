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

def grid_encoding_new(max_len, dim, factor = 10_000, seed = 2147483647):
    position = get_all_positions(max_len)

    i = torch.linspace(0.0, 3.65, dim)
    omega = factor * 0.28 * 1.42 ** i

    generator = torch.Generator().manual_seed(seed)

    thetas = np.pi / 3 * torch.rand(dim, generator=generator)
    r0s = torch.rand(dim, 2, generator=generator) * torch.stack([omega, omega], dim=-1)

    G = hexagon_encoding_new(position, r0s, omega, thetas)

    return G.reshape(max_len, max_len, dim)

if __name__ == "__main__":
    sum_test = False
    sim_test = True
    plot_embed = False

    # test sum behavior
    encodings = grid_encoding_new(28, 128, factor = 100)
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