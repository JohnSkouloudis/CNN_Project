import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from COVID19Dataset import COVID19Dataset
from train_test_functions import train_one_epoch, test,display_confusion_matrix
import torch.optim as optim
import time


#Zhtoumeno 4 
if __name__ == "__main__":


    print
    choice = input('''choose between the following options:
    1.training resNet50 model with the COVID19Dataset
    2.use resNet50 to extract features using only the last layer with the COVID19Dataset:''')

    if choice == '1':
    
        transform = transforms.Compose([
            transforms.Resize((229,229)),
            transforms.ToTensor()
        ])

        dataset = COVID19Dataset(root_dir='COVID-19_Radiography_Dataset',transform=transform)
    
        generator = torch.Generator().manual_seed(42)
        
        train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2], generator=generator)
    
        batch_size = 64
        epochs = 5
    

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")

        model = models.resnet50(pretrained=True)
        model.to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

        lossfn = nn.CrossEntropyLoss()


        start = time.time()

        for epoch in range(epochs):

            print(f'Start of Epoch {epoch+1}/{epochs}')

            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, lossfn, device)

            val_loss, val_accuracy, _ = test(model, val_loader, lossfn, device)

            print(f'End of Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    
    
        print("Results of training resNet50 model:")        
        print(f"Duration of training {epochs} epochs: {time.time()- start} seconds")
        test_loss, test_accuracy, cm = test(model, test_loader, lossfn, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}') 
   
        print(f"Confusion Matrix: {cm}")
        display_confusion_matrix(cm) 


    elif choice == '2':

        resnet50 = models.resnet50(pretrained=True)

        for param in resnet50.parameters():
            param.requires_grad = False

        resnet50.fc = nn.Linear(resnet50.fc.in_features, 4)

        transform = transforms.Compose([
            transforms.Resize((229,229)),
            transforms.ToTensor()
         ])

        dataset = COVID19Dataset(root_dir='COVID-19_Radiography_Dataset',transform=transform)   
        generator = torch.Generator().manual_seed(42)
        
        train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2], generator=generator)
    
        batch_size = 64
        epochs = 5
    

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")

        resnet50.to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(resnet50.fc.parameters(), lr=1e-4, betas=(0.9, 0.99))

        lossfn = nn.CrossEntropyLoss()


        start = time.time()

        for epoch in range(epochs):

            print(f'Start of Epoch {epoch+1}/{epochs}')

            train_loss, train_accuracy = train_one_epoch(resnet50, train_loader, optimizer, lossfn, device)

            val_loss, val_accuracy, _ = test(resnet50, val_loader, lossfn, device)

            print(f'End of Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    
    
        print("Results of training resNet50 model:")        
        print(f"Duration of training {epochs} epochs: {time.time()- start} seconds")
        test_loss, test_accuracy, cm = test(resnet50, test_loader, lossfn, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}') 
   
        print(f"Confusion Matrix: {cm}")
        display_confusion_matrix(cm) 


    else :
        print("Invalid choice. Exiting program.")
        exit(1)
