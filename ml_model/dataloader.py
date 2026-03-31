from torch.utils.data import Dataset
import torch
import numpy as np
import mne


class EEGDataset(Dataset):
    def __init__(self):
        ''' 
        '''


    def __len__(self):
        '''
        '''


    def __getitem__(self, idx):
        '''
        '''


def main():
    '''
    test it
    '''
    data = np.load('dataset.npz')
    X, y = data['X'], data['y']

    dataset = EEGDataset()

if __name__ == "__main__":
    main()