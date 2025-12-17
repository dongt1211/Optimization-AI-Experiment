import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch

train_data= pd.read_csv("./input/fashion-mnist_train.csv")
test_data= pd.read_csv("./input/fashion-mnist_test.csv")

## Dataset Part
class CustomDataset(Dataset):
    
    def __init__(self,dataframe,transform=None):
        self.dataframe=dataframe
        self.transform=transform
        
        
    def __len__(self):
        
        return len(self.dataframe)
    
    def __getitem__(self,idx):
        label = self.dataframe.iloc[idx, 0]
        image_data = self.dataframe.iloc[idx, 1:].values.astype('uint8').reshape((28, 28, 1))  # noqa: E501
        
        if(self.transform):
            image=self.transform(image_data)
            
        return image,label
    

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset=CustomDataset(train_data,transform=transform)
test_dataset=CustomDataset(test_data,transform=transform)

# Save transformed datasets as .pt files
torch.save(train_dataset, './input/preprocessed_dataset/train_dataset.pt')
torch.save(test_dataset, './input/preprocessed_dataset/test_dataset.pt')