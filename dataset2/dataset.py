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
        label_fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/label2/')

        with label_fs.open(os.path.join('workspaceartifactstore/label2', label_dirc), 'r') as f:
            for line in f.readlines():
                line = str(line, 'utf-8').strip().split(' ')
                npy_file = line[0]
                label = int(line[1])
                self.data.append((npy_file, label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        npy_file, label = self.data[index]

        if self.mode == 'train':
            fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/train2/')
        elif self.mode == 'test':
            fs = AzureMachineLearningFileSystem('azureml://subscriptions/xxx/resourcegroups/wang851-rg/workspaces/771/datastores/workspaceartifactstore/paths/test2/')

        img_path = os.path.join('workspaceartifactstore', self.mode + '2', npy_file)

        with fs.open(img_path) as f:
            img_np = np.load(f)
            img = Image.fromarray(img_np.astype('uint8'), 'RGB')
            if self.mode == 'train':
                img_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            elif self.mode == 'test':
                img_transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img = img_transform(img)
            return img, label