import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, n_in,n_filters,stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_filters, kernel_size=3, padding=1, stride=stride)

        self.batchnm1 = nn.BatchNorm2d(n_filters)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)

        self.batchnm2 = nn.BatchNorm2d(n_filters)

        self.relu2 = nn.ReLU()

        self.extra_layer = None

        if stride ==2:
            extra_layer =  nn.Conv2d(in_channels=n_in, out_channels=n_filters, kernel_size=1, stride=2)
                                         
                
    

    def forward(self,x):

        identity = x

        x = self.conv1(x)
        x = self.batchnm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnm2(x)
        x = self.relu2(x)

        if self.extra_layer is not None:
            identity = self.extra_layer(identity)
            x += identity
            x= nn.ReLU(x)
        
        return x

        
