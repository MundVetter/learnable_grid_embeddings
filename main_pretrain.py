import os
import argparse
import torch as tc
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import timm.optim.optim_factory as optim_factory

import utils.dataset as dataset
import utils.misc as misc
from models_mae import MaskedAutoencoderViT
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision import transforms

def train(args):
    device = misc.get_device(args.use_cuda)
    misc.print_which_device(args.use_cuda)

    model = MaskedAutoencoderViT(**vars(args))
    model = model.to(device)

    eff_batch_size = args.batch_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount()

    save_path = os.path.join('save', misc.current_date_and_time())
    os.makedirs(save_path)
    writer = SummaryWriter(logdir=save_path)
    
    dataset_type = getattr(datasets, args.dataset)
    if args.dataset == 'STL10':
        train_data = dataset_type(root=args.data_path, split='unlabeled', download=True, transform=transforms.ToTensor())
        test_data = dataset_type(root=args.data_path, split='test', download=True, transform=transforms.ToTensor())
    else:
        train_data = dataset_type(root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
        test_data = dataset_type(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
    train_data = dataset.FluidMask(train_data, args)
    test_data = dataset.FluidMask(test_data, args)

    # train_data = dataset.MNIST_Glimpses_classify(True, args.pos_encoding, args.fashion)
    # test_data = dataset.MNIST_Glimpses_classify(False, args.pos_encoding, args.fashion)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    for epoch in range(1, args.n_epochs):
        model.train()
        for i, (glimpses, locations, imgs, targets) in enumerate(train_loader):
            misc.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)

            glimpses = glimpses.to(device)
            locations = locations.to(device)
            targets = targets.to(device)
            imgs = imgs.to(device)

            loss, _ = model(glimpses, locations, imgs)
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
            optimizer.zero_grad()
            writer.add_scalar('loss', loss.item(), i + (epoch - 1) * len(train_loader))
            writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], i + (epoch - 1) * len(train_loader))

        # TODO shorter epochs or different saving

        print('{}/{}'.format(str(epoch).zfill(2), args.n_epochs))

        with tc.no_grad():
            model.eval()
            total_loss = 0
            for i, (glimpses, locations, imgs, targets) in enumerate(test_loader):
                glimpses = glimpses.to(device)
                locations = locations.to(device)
                targets = targets.to(device)
                imgs = imgs.to(device)

                optimizer.zero_grad()
                loss, predictions = model(glimpses, locations, imgs)
                total_loss += loss.item()
                if i == 0:
                    img_grid = make_grid(imgs)
                    predictions_grid = make_grid(model.unpatchify(predictions))
                    writer.add_image('images', img_grid, epoch)
                    writer.add_image('predictions', predictions_grid, epoch)

                    input_imgs = []
                    # predicted_imgs = []
                    for glimpse_item, location_item in zip(glimpses,  locations):
                        img = misc.create_image_from_glimpses(glimpse_item, location_item, args.img_size, args.in_chans)
                    #     predicted_img = utils.create_image_from_glimpses(predicted_item, location_item, args.img_size)
                    #     predicted_imgs.append(predicted_img)
                        input_imgs.append(img)
                    # predicted_imgs = torch.stack(predicted_imgs)
                    input_imgs = torch.stack(input_imgs)
                    # predicted_imgs_grid = make_grid(predicted_imgs)
                    input_imgs_grid = make_grid(input_imgs)
                    writer.add_image('input glimpses', input_imgs_grid, epoch)
                    # writer.add_image('predicted glimpses', predicted_imgs_grid, epoch)
            writer.add_scalar('validation loss', total_loss / len(test_loader), epoch)

        # TODO start saving only after a time, or some quick delete
        if epoch % 10 == 0:
            tc.save(model.state_dict(), '{}/model_{}.pt'.format(save_path, str(epoch).zfill(2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=200)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=4)

    parser.add_argument('--encoding_type', choices=['hexagon', 'square', 'triangle'], default='triangle', help='only used if positional_encoding is grid')
    parser.add_argument('--pos_embed', choices=['grid', 'naive', 'none'], default='naive')
    parser.add_argument('--div_factor', type=int, default=100)

    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--n_patches', type=int, default=9)
    parser.add_argument('--dataset', type=str, default="STL10")
    parser.add_argument('--data_path', type=str, default="data_input")

    args = parser.parse_args()
    train(args=args)