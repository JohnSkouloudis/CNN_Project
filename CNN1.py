import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from COVID19Dataset import COVID19Dataset
from torchvision import transforms
import torch.optim as optim
from train_test_functions import train_one_epoch, test,display_confusion_matrix
import time

class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)  
        self.relu1 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(16 * 55 * 55, 32)  
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(32, 4)
    
    def forward(self,x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)


        return x


# Zhtoumeno 2
if __name__ == "__main__":




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

    model = CNN1()
    model.to(device)

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



         






    