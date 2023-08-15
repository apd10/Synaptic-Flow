import torch
import torchvision
import pdb
import copy
import numpy as np



def split_dataloader(dataloader, first_ratio, seed):
    ''' split a given dataset into two datasets '''

    if type(dataloader.dataset) not in [torchvision.datasets.cifar.CIFAR100, torchvision.datasets.cifar.CIFAR10, torchvision.datasets.mnist.MNIST ]:
        raise NotImplementedError
    dataset = dataloader.dataset
    gen1 = torch.Generator().manual_seed(seed)

    total_size = len(dataset)
    first_size = int(first_ratio * total_size)
    second_size = total_size - first_size
    dataset_list = torch.utils.data.random_split(dataset, [first_size, second_size], generator=gen1)
    f=dataset_list[0]
    s=dataset_list[1]

    kwargs = {'num_workers': dataloader.num_workers, 'pin_memory': dataloader.pin_memory}
    first_dataloader = torch.utils.data.DataLoader(dataset=f,
                                             batch_size=dataloader.batch_size, 
                                             shuffle=True,
                                             **kwargs)

    second_dataloader = torch.utils.data.DataLoader(dataset=s,
                                             batch_size=dataloader.batch_size, 
                                             shuffle=True,
                                             **kwargs)



    # test
    #for d in [dataloader, first_dataloader, second_dataloader]:
    #    print(d)
    #    for i,j in enumerate(d):
    #        print(i)

    return first_dataloader, second_dataloader

