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
    return
    

def test(model,dataloader,lossfn,device):
    return



if __name__ == '__main__':
    y_true = torch.tensor([0, 1, 2, 1, 3, 2, 1, 0, 2, 2])
    y_pred = torch.tensor([0, 2, 2, 1, 0, 2, 1, 0, 0, 2])

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    display_confusion_matrix(conf_matrix)

    