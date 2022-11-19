import os
import argparse
import torch as tc
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

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

    save_path = os.path.join('save', utils.current_date_and_time())
    os.makedirs(save_path)
    writer = SummaryWriter(logdir=save_path)

    train_data = dataset.MNIST_Glimpses_classify(train=True)
    test_data = dataset.MNIST_Glimpses_classify(train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

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
            writer.add_scalar('test_accuracy', correct / len(test_data.dataset), epoch)


        # TODO start saving only after a time, or some quick delete

        if epoch % 10 == 0:
            tc.save(model.state_dict(), '{}/model_{}.pt'.format(save_path, str(epoch).zfill(2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mlp_dim', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--no_positional_info', type=bool, default=False)
    args = parser.parse_args()
    train(args=args)