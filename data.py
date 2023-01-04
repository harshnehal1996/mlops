import torch
import numpy as np
from global_vars import isTrain
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, image, label):
        self.images = image
        self.labels = label
        self.transform_probability = 0.2
    
    def __len__(self):
        return len(self.labels)
    
    def rotate(self, tensor):
        if np.random.randint(2):
            return torch.flip(tensor, [0]).T.unsqueeze(0)
        else:
            return torch.flip(tensor, [1]).T.unsqueeze(0)

    def __getitem__(self, idx):
        global isTrain
        
        if isTrain and self.labels[idx].item() != 6 and self.labels[idx].item() != 9 and np.random.uniform() < self.transform_probability:
            return self.rotate(self.images[idx]), self.labels[idx]
        
        return self.images[[idx]], self.labels[idx]


def mnist():
    suffix = [0,1,2,3,4]
    base_path = '/home/harsh/dtu_mlops/data/corruptmnist'
    images = []
    labels = []
    for s in suffix:
        path = base_path + '/train_' + str(s) + '.npz'
        data = np.load(path)
        images.append(torch.FloatTensor(data['images'].astype(np.float32)))
        labels.append(torch.LongTensor(data['labels']))
    
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    path = base_path + '/test' + '.npz'
    data = np.load(path)
    test_images = torch.FloatTensor(data['images'].astype(np.float32))
    test_labels = torch.LongTensor(data['labels'])
    train_dataset = Data(images, labels)
    test_dataset = Data(test_images, test_labels)

    return train_dataset, test_dataset
