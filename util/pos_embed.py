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

def generate_grid_posembed(max_len, dim, temperature=10_000, encode_function=hexagon_encoding):
    """ Generate position encoding for a given max_len and d_model.
    """
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

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, factor=10_000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / factor**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def generate_naive_posembed(max_len, dim, factor):
    """ Generate naive pos embedding"""
    dim = dim // 2
    assert dim % 2 == 0, 'dim must be even'
    position_encoding = tc.zeros(max_len, dim)
    position = tc.arange(0, max_len, dtype=tc.float).unsqueeze(1)
    div_term = tc.exp(tc.arange(0, dim, 2).float() * (-math.log(factor) / dim))
    position_encoding[:, 0::2] = tc.sin(position * div_term)
    position_encoding[:, 1::2] = tc.cos(position * div_term)

    return position_encoding

def dot_product_sim(encoding_grid, position):
    cell = encoding_grid[position[0], position[1], :]

    # calculate dot product between center and all other points
    dot_product = tc.einsum('d, hwd -> hw', cell, encoding_grid)
    # normalize
    dot_product = dot_product / tc.norm(cell) / tc.norm(encoding_grid, dim=-1)
    return dot_product

if __name__ == "__main__":
    # import utils.misc as misc
    # k1 = tc.tensor([1.0, 0.0])
    # k2 = tc.tensor([0.5, 3**0.5/2])

    # # plot the vectors
    # plt.plot([0, k1[0]], [0, k1[1]], 'r')
    # plt.plot([0, k2[0]], [0, k2[1]], 'r')
    # plt.show()

    # sincos = get_1d_sincos_pos_embed_from_grid(8, tc.arange(0, 28, 1))
    # misc.plot_image(sincos)
    # plt.show()

    # print(triangle_encoding(tc.tensor([[0, 500]])))
    sincos_embed = torch.tensor(get_2d_sincos_pos_embed(256, 40)).reshape(40, 40, 256)

    # # plot the encoding
    encodings = generate_grid_posembed(28, 256, temperature = 10_000, encode_function=hexagon_encoding)
    H, W, D = encodings.shape

    pos = H // 2, W // 2
    product = dot_product_sim(encodings, pos)
    plt.imshow(product)
    plt.show()
    # plt.imshow(encodings[:, :, :].reshape(16*16, 8))
    # plt.show()
    # encodings_old = generate_position_encoding_old(100, 4, 10_000,encode_function=hexagon_encoding)
    # plt.imshow(encodings_old)
    # plt.show()

    # x = tc.arange(0, 16, 1)
    # y = tc.arange(0, 16, 1)

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
    # embed_dim = encodings.shape[-1]
    # # t = get_all_positions(224)
    # for i in range(embed_dim):
    #     z = encodings[:, :, i]
    #     ax.plot_wireframe(range(16), range(16), z)
    # plt.title('Hexagon Encoding sin')
    # plt.show()

    # # plot old
    # ax2 = plt.axes(projection='3d')
    # ax2.view_init(azim=0, elev=90)
    # ax2.plot_trisurf(t[:, 0], t[:, 1], z, cmap='viridis', edgecolor='none')
    # plt.title('Hexagon Encoding old')
    # plt.show()