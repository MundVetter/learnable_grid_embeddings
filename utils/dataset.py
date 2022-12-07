import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop

import utils.misc as misc


class FluidMask(Dataset):
    def __init__(self, dataset, args, return_original=True):
        self.dataset = dataset
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        self.return_original = return_original
    
    def __len__(self):
        return len(self.dataset)

    def get_glimpse(self, image, glimpse_location):
        x, y = glimpse_location
        half = self.patch_size // 2

        return crop(image, int(x) - half, int(y) - half, self.patch_size, self.patch_size) 

    def get_glimpses(self, image, locations):
        return torch.stack([self.get_glimpse(image, location) for location in locations])
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # generate n_patches random locations within the image
        locations = torch.randint(0, self.img_size, (self.n_patches, 2)) # TODO: generate locations also on the right edge
        patches = self.get_glimpses(image, locations)

        patches = misc.collapse_last_dim(patches, dim=2)

        if self.return_original:
            return patches, locations, image, label
        else:
            return patches, locations, label

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    dataset = torchvision.datasets.MNIST(
        root='./data_input', train=True, download=True, transform=transforms.ToTensor()
    )
    fluid_mask = FluidMask(dataset, image_size=28, patch_size=6, n_patches=5)

    loader = DataLoader(fluid_mask, batch_size=2, shuffle=True)
    for patches, locations, image, label in loader:
        print(patches.shape)
        print(locations.shape)
        print(image.shape)
        print(label.shape)
        break