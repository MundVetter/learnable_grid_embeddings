import os
import io
import math
import datetime
import PIL.Image
import numpy as np
import torch as tc
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from torch._six import inf
import builtins
import time
import os
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist

from torchvision.transforms.functional import center_crop


def print_which_device(use_cuda):
    if use_cuda and tc.cuda.is_available():
        device_number = tc.cuda.current_device()
        gpu_name = tc.cuda.get_device_name(device_number)
        print('Training on {} device number {} with CUDA version {}.'.format(gpu_name, device_number, tc.version.cuda))
    else:
        print('Training on CPU.')

def get_grid_locations(image_size, patch_size):
    offset = patch_size // 2
    row = tc.arange(offset, image_size, patch_size)
    col = tc.arange(offset, image_size, patch_size)
    grid = tc.stack(tc.meshgrid(row, col), dim=-1).reshape(-1, 2)
    return grid

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

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def add_glimpse_to_image(image, glimpse, location, glimpse_size, channel=1):
    x = location[0]
    y = location[1]
    # convert glimpse to 2d
    # glimpse = glimpse.reshape(glimpse_size, glimpse_size)\
    image[:, x:x + glimpse_size,
    y:y + glimpse_size] = glimpse.reshape(channel, glimpse_size, glimpse_size)

    return image

def create_image_from_glimpses(glimpses, locations, img_size=28, channel=3):
    """Create a single image from a sequence of glimpses and locations."""
    # start with a black image
    glimpse_size = int(math.sqrt(glimpses.shape[1] / channel))
    image = torch.zeros((channel, img_size + glimpse_size, img_size + glimpse_size))
    half = glimpse_size // 2
    # loop over glimpses and locations
    for glimpse, location in zip(glimpses, locations):
        # add the glimpse to the image
        image = add_glimpse_to_image(image, glimpse, location, glimpse_size, channel)
    image = image[:, half:-half, half:-half]
    return image



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


if __name__ == '__main__':
    x = tc.rand(15, 1, 4, 4)
    assert collapse_last_dim(x, dim=1).shape == tc.Size([15 * 16])
    assert collapse_last_dim(x, dim=2).shape == tc.Size([15, 16])
    assert collapse_last_dim(x, dim=3).shape == tc.Size([15, 1, 16])
    assert collapse_last_dim(x, dim=4).shape == tc.Size([15, 1, 4, 4])
    assert collapse_last_dim(x, dim=5).shape == tc.Size([15, 1, 4, 4])

    position_encoding = get_position_embedding(28, 16, 10_000)
    plot_image(position_encoding)
    plt.show()

    # pos_2d = posemb_sincos_2d(28, 16)
    # print(pos_2d.shape)

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
