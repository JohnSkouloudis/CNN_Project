import matplotlib.pyplot as plt
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import os

class COVID19Dataset(Dataset):

    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir,class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path,image_name)
                self.image_paths.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def display_batch(self,indexes):
        n = len(indexes)
        cols = int(n ** 0.5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        for idx,ax in zip(indexes,axes):
            image, label = self[idx]

            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            
            ax.imshow(image)
            ax.set_title(self.classes[label])
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
def plot_bar_diagram(dataset):
    
    class_count = {class_name:0 for class_name in dataset.classes}

    classes = dataset.classes
    labels = dataset.labels

    for label in labels:
        class_name = classes[label]
        class_count[class_name] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(class_count.keys(), class_count.values(), color =['red', 'blue', 'green', 'yellow'])

    plt.xlabel('Classes')
    plt.ylabel('Image Count')
    plt.show()



# Zhtoumeno 1
if __name__ == '__main__':

    img = Image.open('COVID-19_Radiography_Dataset/COVID/COVID-1.png')
    width, height = img.size
    print(f"Image dimensions: {width} x {height}")
    

    dataset = COVID19Dataset('COVID-19_Radiography_Dataset',transform=None)
    random_indexes = random.sample(range(len(dataset.image_paths)),25)
    dataset.display_batch(random_indexes)
    plot_bar_diagram(dataset)
