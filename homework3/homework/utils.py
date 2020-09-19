import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import torchvision
import os


import numpy as np
import csv

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        i = 1
        #print(transform)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(128,128),
                                                         torchvision.transforms.RandomResizedCrop(64),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                       torchvision.transforms.ToTensor()])
        #tr_image = torchvision.transforms.ToTensor()
        #self.transform1 =  torchvision.transforms.Compose([(torchvision.transforms.ToTensor())])
        #print(self.transform1)
        to_image=transforms.ToPILImage()
        #i = 1;
        self.list1 = []
        #print(dataset_path1)
        for filename in sorted(os.listdir(dataset_path)):
            file_path = os.path.abspath(os.path.join(dataset_path, filename))
            _, file_extension = os.path.splitext(file_path)
            if(file_extension == ".jpg"):
                image = Image.open(file_path)
                #image.save(i+".jpg")
                self.list1.append(image)
                image.load()
                

                
        #self.data = torch.stack(list1)
        #list1.clear()
        #print("loaded succesfully")
        #print(self.data.shape)

        csv_path = os.path.join(dataset_path,"labels.csv")
        with open(csv_path,'r') as dest_f:
            data_iter = csv.reader(dest_f,
                           delimiter = ',')
            label = [data1 for data1 in data_iter]
        
        label = np.array(label)
        label = (label[1:,1])
        mod_label = [LABEL_NAMES.index(label_ind) for label_ind in label]
        self.label = torch.from_numpy(np.array(mod_label))
        self.nsamples = len(self.list1)

    def __len__(self):

        return self.nsamples
        raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        img = self.list1[idx]
        return self.transform(img),self.label[idx].item();
        """
        Your code here
        return a tuple: img, label
        """
        raise NotImplementedError('SuperTuxDataset.__getitem__')



class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
