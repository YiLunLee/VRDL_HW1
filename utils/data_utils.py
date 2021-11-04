import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
import os
import torch.utils.data as data
logger = logging.getLogger(__name__)


def get_image_list(path):
    with open(path, 'r') as f:
        data_list = []
        for line in f.readlines():
            words = line.strip().split(' ')
            image = words[0]
            if len(words) > 1:
                label = int(words[1].split('.')[0])
                data_list.append((image, label))
            else:
                data_list.append((image, -1))
    return data_list


class Datasets(data.Dataset):
    def __init__(self, root='', is_train=True, dstype='train', image_list='training_labels.txt', transforms=None):
        self.image_root = os.path.join(root, dstype)
        self.image_list = get_image_list(os.path.join(root, image_list))
        self.size = len(self.image_list)
        self.transforms = transforms
        self.dstype = dstype
        self.is_train = is_train

    def __getitem__(self, index):
        img_name, label = self.image_list[index]
        img_path = os.path.join(self.image_root, img_name)
        img = Image.open(img_path)
        label = torch.LongTensor([label-1])

        if self.transforms is not None:
            img = self.transforms(img)

        if self.is_train or (self.is_train is False and self.dstype == 'train'):
            return img, label
        else:
            return img, img_name

    def __len__(self):
        return self.size


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "CUB_HW":
        train_transform = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = Datasets(root='./dataset', is_train=True, dstype='train', image_list='train_set_labels.txt', transforms=train_transform)
        valset = Datasets(root='./dataset', is_train=False, dstype='train', image_list='val_set_labels.txt', transforms=test_transform)
        testset = Datasets(root='./dataset', is_train=False, dstype='test', image_list='testing_img_order.txt', transforms=test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True) if testset is not None else None
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, val_loader, test_loader
