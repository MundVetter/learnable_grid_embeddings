import torch
import torch.nn as nn
import argparse

import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
import utils
from model import MapFormer_classifier
from main_c import calculate_accuracy

def test(args):
    device = utils.get_device(args.use_cuda)
    utils.print_which_device(args.use_cuda)

    model = MapFormer_classifier(args)
    model = model.to(device)
    model.load_state_dict(torch.load(f'{args.folder}/model_{args.epoch}.pt'))
    model.eval()

    class Rotation:
        def __init__(self, angle) -> None:
            self.angle = angle
        def __call__(self, x):
            return transforms.functional.rotate(x, self.angle)

    transform_sequence = transforms.Compose([
        transforms.ToTensor(),
        Rotation(args.test_rotation)
    ])

    dataset_type = getattr(datasets, args.dataset)
    test_data = dataset_type(root=args.data_path, train=False, download=True, transform=transform_sequence)
    test_data = dataset.FluidMask(test_data, args, return_original=False)

    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    accs = []
    for i in range(10):
        accs.append(calculate_accuracy(model, test_data, device))
    accuracy = np.mean(accs)
    std = np.std(accs)
    print(f"final accuracy: {accuracy} +- {std}")


def get_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mlp_dim', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=28)
    parser.add_argument('--div_factor', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--n_patches', type=int, default=200)
    parser.add_argument('--data_path', type=str, default='data_input')
    parser.add_argument('--pos_encoding', choices=['grid', 'naive', 'lff', 'none'], default='lff')
    parser.add_argument('--encoding_type', choices=['hexagon', 'square', 'triangle', 'hexagon_1', 'hexagon_n14'], default='hexagon', help='only used if positional_encoding is grid')
    parser.add_argument('--rotation', type=int, help="Determines the rotation of the unit vectors in degrees. Only used if positional_encoding is grid", default=4) # TODO: add support for naive
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--cosine', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--H_dim', type=int, default=32)
    parser.add_argument('--F_dim', type=int, default=128)

    parser.add_argument('--test_rotation', type=int, default=3)
    
    parser.add_argument('--folder', type=str, default="MNISTsave\Jan10_12-48_(25,561000)")
    parser.add_argument("--epoch", type=str, default="00")
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    test(args)