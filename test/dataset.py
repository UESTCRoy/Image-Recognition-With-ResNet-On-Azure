import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from azureml.fsspec import AzureMachineLearningFileSystem

class MyDataset(Dataset):
    def __init__(self, label_dirc, mode):
        self.mode = mode
        self.data = []
        if label_dirc is not None:
            label_fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/label/')

            with label_fs.open(os.path.join('workspaceartifactstore/label', label_dirc), 'r') as f:
                for line in f.readlines():
                    line = str(line, 'utf-8').strip().split(' ')
                    npy_file = line[0]
                    label = int(line[1])
                    self.data.append((npy_file, label))
        else:
            if mode == 'test':
                fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/test/')
                for i in range(10000):  # assuming there are 10,000 test images
                    npy_file = f"{i}.npy"
                    self.data.append((npy_file, -1))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        npy_file, label = self.data[index]

        if self.mode == 'train':
            fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/train/')
        elif self.mode == 'test':
            fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/test/')

        img_path = os.path.join('workspaceartifactstore', self.mode, npy_file)

        with fs.open(img_path) as f:
            img_np = np.load(f)
            img = Image.fromarray(img_np.astype('uint8'), 'RGB')
            if self.mode == 'train':
                img_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            elif self.mode == 'test':
                img_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img = img_transform(img)
            return img, label