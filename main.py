import os
import argparse
import torch as tc
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import dataset
import utils
from model import MapFormer


def train(args):
    device = utils.get_device(args.use_cuda)
    utils.print_which_device(args.use_cuda)

    model = MapFormer(args)
    model.train()
    model = model.to(device)

    optimizer = tc.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    save_path = os.path.join('save', utils.current_date_and_time())
    os.makedirs(save_path)
    writer = SummaryWriter(logdir=save_path)

    train_data = dataset.MNIST_Glimpses()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.n_epochs):
        for i, (glimpses, locations, targets, query_locations) in enumerate(train_loader):

            glimpses = glimpses.to(device)
            locations = locations.to(device)
            targets = targets.to(device)
            query_locations = query_locations.to(device)

            inputs = glimpses, locations, query_locations
            predictions = model(inputs)

            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), i + (epoch - 1) * len(train_loader))
            break

        # TODO shorter epochs or different saving

        print('{}/{}'.format(str(epoch).zfill(2), args.n_epochs))

        utils.plot_batch_as_patch(predictions.cpu().detach())
        writer.add_image('0_predictions', utils.plot_to_tensor(), epoch)

        utils.plot_batch_as_patch(targets.cpu())
        writer.add_image('1_targets', utils.plot_to_tensor(), epoch)

        utils.plot_batch_as_patch(glimpses.cpu())
        writer.add_image('2_glimpses', utils.plot_to_tensor(), epoch)

        # TODO start saving only after a time, or some quick delete

        if epoch % 10 == 0:
            tc.save(model.state_dict(), '{}/model_{}.pt'.format(save_path, str(epoch).zfill(2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=36)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    args = parser.parse_args()
    train(args=args)
