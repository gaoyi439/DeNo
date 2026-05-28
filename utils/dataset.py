import re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ComFoldData(Dataset):
    def __init__(self, data, label, com_label, pml_label,train=True):
        # reading data as table form
        self.images = data
        self.labels = label
        self.comp_labels = com_label
        self.pml_labels = pml_label

        self.train = train

    def __getitem__(self, index):
        img, target, comp_label,pml_label = torch.from_numpy(self.images[index]).float(), \
                                  torch.from_numpy(self.labels[index]).float(), \
                                  torch.from_numpy(self.comp_labels[index]).float(),\
                                  torch.from_numpy(self.pml_labels[index]).float()

        return img, target, comp_label, pml_label, index

    def __len__(self):
        return len(self.labels)

'''
fold: fold-th fold of 10-cross fold
nfold: 10
'''
def ComFold(batchsize, Filename, nfold, fold):
    data = np.genfromtxt(Filename[0], delimiter=',')
    label = np.genfromtxt(Filename[1], delimiter=',')
    com_label = np.genfromtxt(Filename[2], delimiter=',')
    pml_label = np.genfromtxt(Filename[3],delimiter=',')
    
    # spliting validation data index
    y = np.arange(len(com_label))
    y, val_index = train_test_split(y, test_size=0.1, random_state=0)

    # the remianing data using to train and test

    n_test = len(y) // nfold
    print('test size:', n_test)

    start = fold * n_test

    if start+n_test > len(com_label):
        test = y[start:]
    else:
        test = y[start:start+n_test]

    train = np.setdiff1d(y, test)

    # Min-max standardization using training statistics only (avoid data leakage)
    eps = 1e-12
    train_x = data[train, :]
    min_x = np.min(train_x, axis=0, keepdims=True)
    max_x = np.max(train_x, axis=0, keepdims=True)
    scale = max_x - min_x
    data = (data - min_x) / (scale + eps)

    # training dataset
    train_scene = ComFoldData(data[train, :], label[train, :], com_label[train, :],pml_label[train,:], train=True)
    print("train data shape:", data[train, :].shape)
    train_loader = DataLoader(
        train_scene,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # Data loader for test dataset
    size = len(test)
    test_scene = ComFoldData(data[test, :], label[test, :], com_label[test, :],pml_label[test,:], train=False)
    print("test data & label shape:", data[test, :].shape, label[test, :].shape)
    test_loader = DataLoader(
        test_scene,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )

    # Data loader for validation dataset
    size = len(test)
    val_data = ComFoldData(data[val_index, :], label[val_index, :], com_label[val_index, :],pml_label[val_index, :],train=False)
    print("validation data & label shape:", data[val_index, :].shape, label[val_index, :].shape)
    val_loader = DataLoader(
        val_data,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, test_loader,val_loader