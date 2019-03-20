# DATA PREPROCESSING CLASS
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DatasetProcessing(Dataset):
	#used to initialise the class variables - transform, data, target
    def __init__(self, data_dir, target, transform=None): 
        self.transform = transform
        self.data_dir = data_dir
        self.target = torch.from_numpy(np.array(target)).float() 
    
	#used to retrieve the X and y index value and return it
    def __getitem__(self, index):
    	img = cv2.imread(self.data_dir[index], 1)
    	img_tensor = self.transform(np.array(img))
    	img_target = torch.tensor(self.target[index], dtype=torch.long)

    	return img_tensor, img_target

    def __len__(self): #returns the length of the data
        return len(list(self.data_dir))