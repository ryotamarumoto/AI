import os
from PIL import Image

from torch.utils import data
from torchvision import transforms as T


class MVTecADtrain(data.Dataset):

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        """Return one image"""
        if index == 0:
            filename = "ok00{}.png".format(index+1)
        elif index < 10:
            filename = "ok00{}.png".format(index)
        elif index < 100:
            filename = "ok0{}.png".format(index)
        else:
            filename = "ok{}.png".format(index)
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return len(os.listdir(self.image_dir))

class MVTecADtest(data.Dataset):

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        """Return one image"""
        if index == 0:
            filename = "test_{}.png".format(index+1)
        else:
            filename = "test_{}.png".format(index)
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return len(os.listdir(self.image_dir))




def return_MVTecAD_loadertrain(image_dir, batch_size=256, train=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.RandomCrop((128, 128)))
    transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.RandomVerticalFlip(p=0.5))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = MVTecADtrain(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return data_loader


def return_MVTecAD_loadertest(image_dir, batch_size=256, train=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.RandomCrop((128, 128)))
    transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.RandomVerticalFlip(p=0.5))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = MVTecADtest(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return data_loader