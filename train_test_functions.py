import torch
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def confusion_matrix(y,y_pred):

   

    num_classes = torch.unique(torch.cat((y,y_pred)))

    num_classes = len(num_classes)

    cm = torch.zeros(num_classes,num_classes)

    for true,pred in zip(y,y_pred):
        cm[true,pred] += 1

    return cm

def display_confusion_matrix(cm):

    if isinstance(cm,torch.Tensor):
        cm = cm.numpy()

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


def train_one_epoch(model,dataloader,optimizer,lossfn,device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    size = len(dataloader.dataset)


    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = lossfn(pred, y)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)


        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    avg_loss = total_loss / size
    accuracy = correct / total

    return avg_loss,accuracy
    

def test(model,dataloader,lossfn,device):
    model.eval()

    return_loss = 0.0
    return_accuracy = 0.0

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    cm = torch.zeros(4,4)

    with torch.no_grad():
        for X,y in dataloader:

            X,y = X.to(device),y.to(device)

            pred = model(X)
           
            return_loss += lossfn(pred,y).item()
            return_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            if len(torch.unique(torch.cat((y,pred.argmax(1))))) < 4:
                for true,pred in zip(y,pred.argmax(1)):
                    cm[true,pred] += 1
            else:
                cm += confusion_matrix(y,pred.argmax(1))
            


    return_loss /= num_batches
    return_accuracy /= size        

    return return_loss,return_accuracy,cm







    