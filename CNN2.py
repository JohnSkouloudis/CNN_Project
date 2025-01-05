import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from COVID19Dataset import COVID19Dataset
from torchvision import transforms
import torch.optim as optim
from train_test_functions import train_one_epoch, test,display_confusion_matrix
import time

class CNN2(nn.Module):

    def __init__(self , num_classes=4):

         

    
        #1
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()

        #2
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        #3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        #4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()

        #6
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #7
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()

        #8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #9
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()

        #10
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #11
        self.flatten = nn.Flatten()

        #12
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.relu11 = nn.ReLU()

        #13
        self.fc2 = nn.Linear(1024, num_classes)
        
        
    def forward(self,x):

        

        
        #1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        #2
        x = self.pool1(x)

        #3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        #4
        x = self.pool2(x)

        #5
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)

        #6
        x = self.pool3(x)

        #7
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)

        #8
        x = self.pool4(x)

        #9
        x = self.conv10(x)
        x = self.relu10(x)

        #10
        x = self.pool5(x)

        #11
        x = self.flatten(x)

        #12
        x = self.fc1(x)
        x = self.relu11(x)

        #13
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)

        return x
    
        

#Zhtoumeno 3
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((229,229)),
        transforms.ToTensor()
    ])

    

    dataset = COVID19Dataset(root_dir='COVID-19_Radiography_Dataset',transform=transform)

    generator = torch.Generator().manual_seed(42)
    
    train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    batch_size = 64
    epochs = 20
    patience = 5
    early_stop_threshhold = 0.5
    counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")


    model = CNN2().to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    lossfn = nn.CrossEntropyLoss()


    start = time.time()

    for epoch in range(epochs):

        print(f'Start of Epoch {epoch+1}/{epochs}')

        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, lossfn, device)

        val_loss, val_accuracy, _ = test(model, val_loader, lossfn, device)

        print(f'End of Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if abs(train_loss - val_loss) > early_stop_threshhold:
            counter += 1
        
        else:
            counter = 0

        if counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    print(f"Duration of training {epochs} epochs: {time.time()- start} seconds")
    test_loss, test_accuracy, cm = test(model, test_loader, lossfn, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}') 
   
    print(f"Confusion Matrix: {cm}")
    display_confusion_matrix(cm)  

