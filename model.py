import torch.nn as nn
import torch.nn.init as init

class BasicBlock(nn.Module):
    expansion=1
    
    def __init__(self,in_planes,planes,stride):
        super(BasicBlock,self).__init__()
        
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        
        self.shortcut=nn.Sequential()
        if((stride!=1)or(in_planes!=self.expansion*planes)):
            self.shortcut=nn.Sequential(nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                                       nn.BatchNorm2d(self.expansion*planes))
            
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=nn.ReLU()(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=out+self.shortcut(x)
        out=nn.ReLU()(out)
        
        return out  
    

class ResNet(nn.Module):
    
    def __init__(self,block,num_blocks,num_classes=10):
        super(ResNet,self).__init__()
        
        self.in_planes=64
        self.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)
        
        for m in self.modules():
            if(isinstance(m,nn.Conv2d)):
                init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                
            elif(isinstance(m,nn.BatchNorm2d)):
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
                
    def _make_layer(self,block,planes,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

