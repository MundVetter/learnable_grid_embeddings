import os
import argparse
import torch as tc
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from torchvision import datasets
from torchvision import transforms

import dataset
import utils
from model import MapFormer_classifier

from utils import calculate_accuracy

from rotations_test import test_only


def train(args):
    device = utils.get_device(args.use_cuda)
    utils.print_which_device(args.use_cuda)

    model = MapFormer_classifier(args)
    model.train()
    model = model.to(device)
    print("Pos encoding:", args.pos_encoding)
    print("Embed size:", args.d_model)
    if args.pos_encoding == 'grid':
        print("Grid type:", args.encoding_type)
    print("N patches:", args.n_patches)
    print("Batch size:", args.batch_size)
    print("Rotation:", args.rotation)
    print("H:", args.H_dim)
    print("F:", args.F_dim)

    optimizer = tc.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    save_path = os.path.join(f'MNISTsave_{args.pos_encoding}', utils.current_date_and_time())
    os.makedirs(save_path)
    writer = SummaryWriter(logdir=save_path)

    dataset_type = getattr(datasets, args.dataset)

    train_data = dataset_type(root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = dataset_type(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
    
    train_data = dataset.FluidMask(train_data, args, return_original=False)
    test_data = dataset.FluidMask(test_data, args, return_original=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    accuracy = 0
    for epoch in range(1, args.n_epochs):
        model.train()
        for i, (glimpses, locations, targets) in enumerate(train_loader):

            glimpses = glimpses.to(device)
            locations = locations.to(device)
            targets = targets.to(device)

            inputs = glimpses, locations
            predictions = model(inputs)

            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), i + (epoch - 1) * len(train_loader))

        # TODO shorter epochs or different saving

        print('{}/{}'.format(str(epoch).zfill(2), args.n_epochs))
        accuracy = calculate_accuracy(model, test_data, device)
        writer.add_scalar('test_accuracy', accuracy, epoch)
        print(f"test accuracy: {accuracy*100:.2f}", flush=True)


        # TODO start saving only after a time, or some quick delete
        if epoch % 10 == 0:
            tc.save(model.state_dict(), '{}/model_{}.pt'.format(save_path, str(epoch).zfill(2)))

    print("Calculating final accuracy")
    accs = []
    for i in range(10):
        accs.append(calculate_accuracy(model, test_data, device))
    accuracy = np.mean(accs)
    std = np.std(accs)
    print(f"final accuracy: {accuracy} +- {std}")

    test_only(model, args)

    return accuracy

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

    parser.add_argument('--test_rotation', type=int, default=10)

    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args=args)
