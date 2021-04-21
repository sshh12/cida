from torch.utils.data import Dataset
import torchvision
import torch

import numpy as np


class RotatedMNIST(Dataset):
    def __init__(self, data_dir, rotate_range=(0, 360), train_range=(0, 45)):
        self.data, self.targets = torch.load(data_dir)
        self.rotate_range = rotate_range
        self.train_range = train_range

    def __getitem__(self, index):
        from PIL import Image

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        rot_min, rot_max = self.rotate_range
        angle = np.random.rand() * (rot_max - rot_min) + rot_min

        img = torchvision.transforms.functional.rotate(img, angle)
        img = torchvision.transforms.ToTensor()(img).to(torch.float)

        train_min, train_max = self.train_range
        is_train = angle >= train_min and angle <= train_max

        return img, target, np.array([angle / 360.0], dtype=np.float32), is_train

    def __len__(self):
        return len(self.data)

    @staticmethod
    def download(root_dir="data"):
        from torchvision.datasets import MNIST

        MNIST(root=root_dir, download=True)

    @staticmethod
    def domains_to_labels(domain):
        return np.array(["{}-{}°".format(int(angle[0] * 8) * 45, int(angle[0] * 8 + 1) * 45) for angle in domain])