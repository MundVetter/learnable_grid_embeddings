import torch as tc
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop

import utils


class GlimpseSequence:
    def __init__(self, image_size, glimpse_size, shift_size):
        self.image_size = image_size
        self.glimpse_size = glimpse_size
        assert 2 * shift_size <= image_size - glimpse_size, "Shift size so large, that sampling may be outside " \
                                                            "boundaries. Please choose a smaller value for the shift " \
                                                            "size or glimpse size or both. "
        self.shift_size = shift_size
        self.glimpse_range = image_size - self.glimpse_size
        self.glimpse_location = self.get_random_glimpse_location()

    def get_glimpse(self, image, glimpse_location):
        x, y = glimpse_location

        return crop(image, int(x), int(y), self.glimpse_size, self.glimpse_size)  # left, upper, right, lower
        # return image.crop((x, y, x + self.glimpse_size, y + self.glimpse_size))

    def get_random_glimpse_location(self):
        return tc.randint(0, self.glimpse_range, (2,))

    def get_random_shift(self):
        return tc.randint(-self.shift_size, self.shift_size, (2,))

    def is_in_range(self, number):
        return 0 <= number <= self.glimpse_range

    def update_location(self, old_location):
        shift = self.get_random_shift()
        # bounce off the walls if shift would go beyond
        if self.is_in_range(old_location[0] + shift[0]):
            new_location_0 = old_location[0] + shift[0]
        else:
            new_location_0 = old_location[0] - shift[0]
        if self.is_in_range(old_location[1] + shift[1]):
            new_location_1 = old_location[1] + shift[1]
        else:
            new_location_1 = old_location[1] - shift[1]

        new_location = [new_location_0, new_location_1]
        return new_location

    def get_random_locations(self, sequence_length):
        return tc.stack([self.get_random_glimpse_location() for _ in range(sequence_length)])

    def get_glimpses(self, image, locations):
        return tc.stack([self.get_glimpse(image, location) for location in locations])

    def get_sequence(self, image, sequence_length):
        locations = self.get_random_locations(sequence_length)
        glimpses = self.get_glimpses(image, locations)
        return glimpses, locations

    def get_sequence_with_required_coverage(self, image, sequence_length, required_coverage=0.65):
        locations = self.get_locations_with_required_coverage(sequence_length, required_coverage)
        glimpses = self.get_glimpses(image, locations)
        return glimpses, locations

    def get_locations_with_required_coverage(self, sequence_length, required_coverage=0.65):
        while True:
            locations = self.get_random_locations(sequence_length)
            if self.get_coverage(locations) > required_coverage:
                return locations

    def get_coverage(self, locations):
        # calculate coverage of sequence of glimpses over image
        covered_pixels = set()
        for i, j in locations:
            for k in range(self.glimpse_size):
                for l in range(self.glimpse_size):
                    covered_pixels.add((i + k, j + l))
        coverage = len(covered_pixels) / self.image_size ** 2
        return coverage


# class LocationsDataset(Dataset):

class MNIST_Glimpses(Dataset):
    def __init__(self):
        image_size = 28
        glimpse_size = 4
        shift_size = 4
        self.glimpse_size = glimpse_size
        self.imag_size = image_size
        self.sequence_length = 40
        self.required_coverage = 0.7
        self.image_dataset = datasets.MNIST(root='./data_input', train=True, download=True,
                                            transform=transforms.ToTensor())
        self.locations_dataset_path = 'data_output/locations.pt'
        self.locations = tc.load(self.locations_dataset_path)
        self.glimpser = GlimpseSequence(image_size=image_size, glimpse_size=glimpse_size, shift_size=shift_size)

    def get_locations(self):
        random_idx = int(tc.randint(len(self.locations), (1,)))
        return self.locations[random_idx]

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        locations = self.get_locations()
        row = tc.arange(0, self.imag_size, self.glimpse_size)
        col = tc.arange(0, self.imag_size, self.glimpse_size)
        query_locations = tc.stack(tc.meshgrid(row, col), dim=-1).reshape(-1, 2)
        # locations = self.glimpser.get_locations_with_required_coverage(self.sequence_length, self.required_coverage)
        # query_locations = self.glimpser.get_random_locations(self.sequence_length)

        glimpses = self.glimpser.get_glimpses(image, query_locations)
        targets = self.glimpser.get_glimpses(image, query_locations)



        glimpses = utils.collapse_last_dim(glimpses, dim=2)
        targets = utils.collapse_last_dim(targets, dim=2)

        return glimpses, query_locations, targets, query_locations

class MNIST_Glimpses_classify(Dataset):
    def __init__(self, train=True):
        image_size = 28
        glimpse_size = 4
        shift_size = 4
        self.glimpse_size = glimpse_size
        self.imag_size = image_size
        self.sequence_length = 40
        self.required_coverage = 0.7
        self.image_dataset = datasets.MNIST(root='./data_input', train=train, download=True,
                                            transform=transforms.ToTensor())
        self.locations_dataset_path = 'data_output/locations.pt'
        self.locations = tc.load(self.locations_dataset_path)
        self.glimpser = GlimpseSequence(image_size=image_size, glimpse_size=glimpse_size, shift_size=shift_size)

    def get_locations(self):
        random_idx = int(tc.randint(len(self.locations), (1,)))
        return self.locations[random_idx]

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, targets = self.image_dataset[idx]
        locations = self.get_locations()
        # row = tc.arange(0, self.imag_size, self.glimpse_size)
        # col = tc.arange(0, self.imag_size, self.glimpse_size)
        # query_locations = tc.stack(tc.meshgrid(row, col), dim=-1).reshape(-1, 2)
        # locations = self.glimpser.get_locations_with_required_coverage(self.sequence_length, self.required_coverage)
        # query_locations = self.glimpser.get_random_locations(self.sequence_length)

        glimpses = self.glimpser.get_glimpses(image, locations)



        glimpses = utils.collapse_last_dim(glimpses, dim=2)

        return glimpses, locations, targets



# def make_dataset():
#     import utils
#     image_size = 28
#     glimpse_size = 6
#     shift_size = 6
#     sequence_length = 20
#     required_coverage = 0.7
#     n_images = 10
#     save_folder = 'mnistmap'
#     train_data = datasets.MNIST(root='./data_input', train=True, download=True, transform=transforms.ToTensor())
#     train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#     for i, (image, label) in enumerate(train_loader):
#         glimpser = GlimpseSequence(image_size=image_size, glimpse_size=glimpse_size, shift_size=shift_size)
#         glimpses, locations = glimpser.get_sequence_with_required_coverage(image, sequence_length, required_coverage)
#         # utils.plot_path(image[0, 0], locations)
#         # plt.show()
#         # break


def make_locations_data():
    utils.makedirs('data_output')
    required_coverage = 0.7
    image_size = 28
    glimpse_size = 4
    shift_size = 4
    sequence_length = 40
    glimpser = GlimpseSequence(image_size=image_size, glimpse_size=glimpse_size, shift_size=shift_size)
    n_locations = 10000
    locations = []
    for i in range(n_locations):
        location = glimpser.get_locations_with_required_coverage(sequence_length=sequence_length,
                                                                 required_coverage=required_coverage)
        locations.append(location)
        if i % 100 == 0:
            print(i)
    locations = tc.stack(locations)
    print(locations.shape)
    save_path = 'data_output/locations.pt'
    tc.save(locations, save_path)


if __name__ == '__main__':
    # make_dataset()
    make_locations_data()
