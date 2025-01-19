# CNN-VGG-YOLOV5-102Flowers

This repository is developed as part of Computational Learning Assessment 4 and focuses on the classification of the Oxford 102 Category Flower Dataset using Convolutional Neural Networks (CNN), specifically VGG19 and YOLOv5 architectures.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview

The project aims to implement and compare the performance of VGG19 and YOLOv5 models in classifying images from the Oxford 102 Category Flower Dataset. The dataset comprises 102 flower categories with a total of 8,189 images. By leveraging pre-trained models and fine-tuning them on this dataset, the project seeks to achieve high classification accuracy.

## Repository Structure

The repository is organized as follows:

- **`VGG.py`**: Contains the implementation of the VGG19 model for flower classification.

- **`YOLOV5.py`**: Includes the implementation of the YOLOv5 model adapted for flower classification.

- **`VGG_metrics.csv`**: Stores the training and validation metrics for the VGG19 model.

- **`YOLOV5_metrics.csv`**: Contains the training and validation metrics for the YOLOv5 model.

- **`VGG Plot.png`**: Visual representation of the VGG19 model's performance metrics.

- **`YOLOV5 Plot.png`**: Visual representation of the YOLOv5 model's performance metrics.

- **`vgg19_flower_classifier.pth`**: Saved weights of the fine-tuned VGG19 model.

- **`yolov5_classifier.pt`**: Saved weights of the fine-tuned YOLOv5 model.

- **`README.md`**: Provides an overview and instructions for the project.

- **`.gitignore`**: Specifies files and directories to be ignored by Git.

- **`.gitattributes`**: Defines attributes for pathnames to customize repository behavior.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher

- PyTorch

- Torchvision

- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/liran121211/CNN-VGG-YOLOV5-102Flowers.git
   ```

### Usage
1. Training the VGG19 Model
To train the VGG19 model on the flower dataset:
```bash
python VGG.py
```

2. Training the YOLOv5 Model
To train the YOLOv5 model on the flower dataset:
```bash
python YOLOV5.py
```

### Results
The performance metrics for both models are stored in VGG_metrics.csv and YOLOV5_metrics.csv. Visual representations of the training progress and accuracy are available in VGG Plot.png and YOLOV5 Plot.png.

### References

Oxford 102 Category Flower Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

YOLOv5 by Ultralytics: https://github.com/ultralytics/yolov5

VGG19 Pre-trained Model for Keras: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
