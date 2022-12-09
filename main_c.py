import os
import argparse
import torch as tc
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import dataset
import utils
from model import MapFormer_classifier


def train(args):
    device = utils.get_device(args.use_cuda)
    utils.print_which_device(args.use_cuda)

    model = MapFormer_classifier(args)
    model.train()
    model = model.to(device)

    optimizer = tc.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    save_path = os.path.join('MNISTsave', utils.current_date_and_time())
    os.makedirs(save_path)
    writer = SummaryWriter(logdir=save_path)

    dataset_type = getattr(datasets, args.dataset)

    train_data = dataset_type(root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = dataset_type(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
    
    train_data = dataset.FluidMask(train_data, args, return_original=False)
    test_data = dataset.FluidMask(test_data, args, return_original=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
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

        correct = 0
        with tc.no_grad():
            model.eval()
            for i, (glimpses, locations, targets) in enumerate(test_data):
                glimpses = glimpses.to(device)
                locations = locations.to(device)
                targets = targets.to(device)

                inputs = glimpses, locations
                predictions = model(inputs)

                loss = criterion(predictions, targets)
                correct += (predictions.argmax(dim=1) == targets).sum().item()
            accuracy = correct / len(test_data.dataset)
            writer.add_scalar('test_accuracy', accuracy, epoch)
            print(f"test accuracy: {accuracy:.2f}", flush=True)


        # TODO start saving only after a time, or some quick delete

        if epoch % 10 == 0:
            tc.save(model.state_dict(), '{}/model_{}.pt'.format(save_path, str(epoch).zfill(2)))
    print(f"final accuracy: {accuracy}")
    return accuracy

def get_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--use_cuda', type=bool, default=True)
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
    parser.add_argument('--pos_encoding', choices=['grid', 'naive', 'none'], default='naive')
    parser.add_argument('--encoding_type', choices=['hexagon', 'square', 'triangle'], default='hexagon', help='only used if positional_encoding is grid')
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args=args)