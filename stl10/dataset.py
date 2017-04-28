import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from IPython import embed
import os

def get(batch_size, data_root='/mnt/local0/public_dataset/pytorch/', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'stl10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building STL10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds

if __name__ == '__main__':
    train_ds, test_ds = get(200, num_workers=1)
    for data, target in train_ds:
        print("~~")
