# Machine Learning Assignment - Convolutional Neural Networks (CNNs)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tasks Overview](#tasks-overview)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Useful Links](#useful-links)

## Introduction
This project is part of the Machine Learning course, focusing on Convolutional Neural Networks (CNNs) for classifying chest X-ray images. The dataset used contains images classified into different categories, and multiple models are trained to evaluate performance.

## Dataset
The dataset consists of chest X-ray images categorized as:
- Normal
- Bacterial Pneumonia
- Viral Pneumonia
- COVID-19

The dataset is loaded and preprocessed using a custom PyTorch `Dataset` class.

## Project Structure
- **COVID19dataset.py** – Contains `COVID19Dataset` class and Task 1  
- **test_train_functions.py** – Includes `train_one_epoch`, `test`, `confusion_matrix`, and `display_confusion_matrix`  
- **CNN1.py** – Implements CNN1 and Task 2  
- **CNN2.py** – Implements CNN2 and Task 3  
- **ResNet50.py** – Implements Task 4 using ResNet50  
- **BasicBlock.py** – Implements the bonus task  
- **README.md** – This file

## Tasks Overview

### Task 1: Dataset Exploration
- **File:** `COVID19dataset.py`
- Implements the `COVID19Dataset` class.
- Generates a bar chart showing class distribution.
- Uses `display_batch` to visualize sample images.
- **Observation:** The dataset is imbalanced, with "Normal" images dominating.

### Task 2: Training a Simple CNN (CNN1)
- **File:** `CNN1.py`
- A basic CNN model is implemented and trained for 20 epochs (with early stopping).
- **Results:** The model is relatively good at detecting "Viral Pneumonia" but has moderate accuracy for other classes. False positives are present, which can be problematic in clinical settings.

### Task 3: Training a Deeper CNN (CNN2)
- **File:** `CNN2.py`
- A deeper CNN architecture is implemented.
- **Comparison with CNN1:** CNN2 achieves higher accuracy than CNN1 due to its increased depth and filter count, enabling better feature extraction.

### Task 4: Transfer Learning with ResNet50
- **File:** `ResNet50.py`
- Fine-tunes a pre-trained ResNet50 model.
- **Training Duration:** ResNet50 took approximately 4.7 hours to train.
- **Additional Experiment:** The last classification layer was trained separately.

### Bonus Task: Implementing a Custom BasicBlock
- **File:** `BasicBlock.py`
- A custom `BasicBlock` module is implemented, similar to ResNet blocks.

## Requirements
Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

Install dependencies using:
```sh
pip install torch torchvision numpy matplotlib scikit-learn
```
## How to Run
* make sure to change the root_dir parameter if the directory with the dataset is not in the same directory as the project
  
### Task 1
```sh
python COVID19dataset.py
```
### Task 2
```sh
python CNN1.py
```
### Task 3
```sh
python CNN2.py
```
### Task 4
```sh
python ResNet50.py
```

## Useful Links
🔗 **PyTorch  Resources**  
- 📌 **PyTorch Dataset Tutorial** – Learn how to create a custom dataset in PyTorch  
  [🔗 Creating a Custom Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
- 📌 **PyTorch Neural Network (nn) Module** – Official documentation for PyTorch's neural network layers  
  [🔗 PyTorch nn Module](https://pytorch.org/docs/stable/nn.html)
- 📌 **PyTorch Tensors** – Official documentation for PyTorch's tensors 
  [🔗 PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html)
