import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import ResNet, BasicBlock
from memory_profiler import profile
from dataset import CustomDataset
import pandas as pd
from torchvision import transforms
from torch.amp import autocast, GradScaler
import argparse

parser = argparse.ArgumentParser(description='Training script with configurable parameters') # noqa: E501
parser.add_argument('--using_gpu', type=bool, default=True, help='Use GPU for training')
parser.add_argument('--prerocessed_dataset', type=bool, default=True, help='Use preprocessed dataset')  # noqa: E501
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training') # noqa: E501
parser.add_argument('--parrallel_dataloader', type=bool, default=True, help='Use parallel DataLoader') # noqa: E501
parser.add_argument('--number_of_workers', type=int, default=1, help='Number of DataLoader workers') # noqa: E501
parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training') # noqa: E501
parser.add_argument('--optimizer', type=str, default='Adam', choices=['AdamW', 'SGD', 'Adam'], help='Optimizer type') # noqa: E501
args = parser.parse_args()

using_gpu = args.using_gpu
prerocessed_dataset = args.prerocessed_dataset
batch_size = args.batch_size
parrallel_dataloader = args.parrallel_dataloader
number_of_workers = args.number_of_workers
mixed_precision = args.mixed_precision
optimizer = args.optimizer
if using_gpu: 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if prerocessed_dataset: 
    # Load preprocessed datasets
    train_dataset = torch.load(
        './input/preprocessed_dataset/train_dataset.pt',
        weights_only=False
    )
else:
    train_data= pd.read_csv("./input/fashion-mnist_train.csv")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset=CustomDataset(train_data,transform=transform)

if parrallel_dataloader:
    ## DataLoader Part
   train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, 
                           num_workers= number_of_workers, pin_memory=True)
else:
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

if mixed_precision:
    scaler = GradScaler()




## Train-Validation Split
train_size=int(0.8*len(train_dataset))
valid_size=len(train_dataset)-train_size

train_dataset,valid_dataset=random_split(train_dataset,[train_size,valid_size])




@profile
def train(model,train_loader,optimizer,criterion):
    model.train()
    train_loss=0
    correct=0
    total=0
    
    for images,labels in train_loader:
        images,labels =images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        if mixed_precision:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        train_loss+=loss.detach()
        _,predicted=outputs.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum()
        
    train_accuracy=100*correct/total
    train_loss/=len(train_loader)
    return train_loss,train_accuracy




if __name__ == "__main__":
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    epochs = 1
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'AdamW':
        optimizer = optim.AdamW(resnet18.parameters(),lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(resnet18.parameters(),lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(resnet18.parameters(),lr=learning_rate)

    train_accuracy=[]
    validation_accuracy=[]
    train_losses=[]
    validation_losses=[]

    for epoch in range(epochs):
        train_loss, train_acc = train(resnet18, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.2f}%")  # noqa: E501

