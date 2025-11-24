import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_data(args):
    '''Obtain data
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
    elif args.data == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        trainset = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    return trainloader, testloader