import os
import io
import math
import datetime
import PIL.Image
import numpy as np
import torch as tc
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor


def print_which_device(use_cuda):
    if use_cuda and tc.cuda.is_available():
        device_number = tc.cuda.current_device()
        gpu_name = tc.cuda.get_device_name(device_number)
        print('Training on {} device number {} with CUDA version {}.'.format(gpu_name, device_number, tc.version.cuda))
    else:
        print('Training on CPU.')


def get_position_embedding(max_len, dim, factor):
    dim = dim // 2
    assert dim % 2 == 0, 'dim must be even'
    position_encoding = tc.zeros(max_len, dim)
    position = tc.arange(0, max_len, dtype=tc.float).unsqueeze(1)
    div_term = tc.exp(tc.arange(0, dim, 2).float() * (-math.log(factor) / dim))
    position_encoding[:, 0::2] = tc.sin(position * div_term)
    position_encoding[:, 1::2] = tc.cos(position * div_term)
    return position_encoding


def current_date_and_time():
    now = datetime.datetime.now()
    month = now.strftime("%B")
    times = [now.day, now.hour, now.minute, now.second]
    times = [str(time).zfill(2) for time in times]
    return '{}{}_{}-{}-{}'.format(month[0:3], *times)


def collapse_last_dim(x, dim=3):
    if dim >= len(x.shape):
        return x
    else:
        new_shape = list(x.shape[:(dim - 1)])
        new_shape.append(-1)
        return x.reshape(new_shape)


def reshape_to_patch(x):
    assert len(x.shape) < 4
    patch_size = int(np.sqrt(x.shape[-1]))
    return x.reshape(x.shape[:-1] + (patch_size, patch_size))


def plot_image(image):
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.imshow(image, 'gray')


def plot_to_tensor():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg')
    buffer.seek(0)
    image = PIL.Image.open(buffer)
    tensor = ToTensor()(image)
    plt.close()
    return tensor

# def sort_batch(batch_of_


# def plot_batch(batch_of_images):
#     plot_image(make_grid(batch_of_images, 4)[0])

def plot_batch(batch_of_images):
    n_images = batch_of_images.shape[0]
    approx_sqrt = math.ceil(np.sqrt(n_images))
    fig, axs = plt.subplots(approx_sqrt, approx_sqrt)
    for i, ax in enumerate(list(axs.reshape(-1))):
        if i < len(batch_of_images):
            image = batch_of_images[i, 0]  # second dimension is assumed as batch size
            ax.imshow(image, 'gray')
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)


def plot_batch_as_patch(batch_of_images):
    return plot_batch(reshape_to_patch(batch_of_images))


def plot_square(location, glimpse_size, c):
    edges = [location, location + np.array([0, glimpse_size]),
             location + np.array([glimpse_size, glimpse_size]),
             location + np.array([glimpse_size, 0]), location]
    edges = tc.stack(edges)
    plt.plot(edges[:, 0], edges[:, 1], c=c)


def plot_path(image, locations, glimpse_size=6):
    plot_image(image)
    for i, location in enumerate(locations):
        # c = i / len(locations)
        colors = plt.cm.hsv(np.linspace(0, 1, len(locations)))
        c = colors[i]
        plot_square(location, glimpse_size, c)

def combine_patches_into_image(patches, image_size):
    n_patches = patches.shape[0]
    patch_size = int(np.sqrt(patches.shape[-1]))
    n_rows = int(np.sqrt(n_patches))
    n_cols = n_rows
    image = np.zeros((image_size, image_size))
    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i * n_rows + j]
            patch = patch.reshape(patch_size, patch_size)
            image[i * patch_size:(i + 1) * patch_size,
            j * patch_size:(j + 1) * patch_size] = patch
    return image

def plot_batch_of_patches_as_image(batch_of_patches, image_size=28):
    n_images = batch_of_patches.shape[0]
    approx_sqrt = math.ceil(np.sqrt(n_images))
    fig, axs = plt.subplots(approx_sqrt, approx_sqrt)
    for i, ax in enumerate(list(axs.reshape(-1))):
        if i < len(batch_of_patches):
            patches = batch_of_patches[i]
            image = combine_patches_into_image(patches, image_size)
            ax.imshow(image, 'gray')
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)

def makedirs(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def get_device(use_cuda):
    if use_cuda:
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    return tc.device(device_name)


if __name__ == '__main__':
    x = tc.rand(15, 1, 4, 4)
    assert collapse_last_dim(x, dim=1).shape == tc.Size([15 * 16])
    assert collapse_last_dim(x, dim=2).shape == tc.Size([15, 16])
    assert collapse_last_dim(x, dim=3).shape == tc.Size([15, 1, 16])
    assert collapse_last_dim(x, dim=4).shape == tc.Size([15, 1, 4, 4])
    assert collapse_last_dim(x, dim=5).shape == tc.Size([15, 1, 4, 4])

    position_encoding = get_position_embedding(28, 16, 100)
    plot_image(position_encoding)
    plt.show()

    # encoding = grid_encoding(2, 1)
    # print(encoding)
    


    # plot the vectors
    # k1 = tc.tensor([1.0, 0.0])
    # k2 = tc.tensor([-1/2, 3**0.5/2])
    # k3 = tc.tensor([-1/2, -3**0.5/2])
    # plt.plot([0, k1[0]], [0, k1[1]], 'r')
    # plt.plot([0, k2[0]], [0, k2[1]], 'g')
    # plt.plot([0, k3[0]], [0, k3[1]], 'b')
    # plt.show()
