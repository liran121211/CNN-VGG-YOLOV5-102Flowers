import os

import pandas as pd
import scipy.io
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


class FlowerClassifier:
    def __init__(self, image_dir, label_path, num_classes=102, lr=0.001, batch_size=32, epochs=20):
        self.image_dir = image_dir
        self.label_path = label_path
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self._load_data()

        # Load model
        self.model = self._load_vgg_model()

    def _load_data(self):
        # Load labels
        mat_data = scipy.io.loadmat(self.label_path)
        labels = mat_data['labels'][0] - 1

        # Load all image paths
        image_paths = [
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

        assert len(image_paths) == len(labels), "Mismatch between number of images and labels!"

        # Create a dataset
        dataset = FlowerDataset(image_paths, labels, transform=self.train_transforms)

        # Split into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(0.5 * total_size)
        val_size = int(0.25 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset,
                                                                               [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def _load_vgg_model(self):
        vgg19 = models.vgg19(pretrained=True)
        for param in vgg19.parameters():
            param.requires_grad = False

        vgg19.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes),
            nn.LogSoftmax(dim=1)
        )

        return vgg19.to(self.device)

    def train(self):
        # Define metrics storage
        train_losses = []
        val_losses = []
        val_accuracies = []

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Record training loss
            train_losses.append(running_loss / len(self.train_loader))

            # Validation step
            val_loss, val_accuracy = self.validate(criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{self.epochs}], "
                  f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # After training, plot the graphs
        self._plot_metrics(train_losses, val_losses, val_accuracies)
        self.save_metrics_to_csv(train_losses, val_losses, val_accuracies)

    def validate(self, criterion):
        self.model.eval()
        val_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.forward(inputs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.exp(outputs)
                top_class = probs.max(dim=1)[1]
                equals = top_class == labels
                accuracy += equals.float().mean().item()

        return val_loss / len(self.val_loader), accuracy / len(self.val_loader)

    def test(self):
        self.model.eval()
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.forward(inputs)
                probs = torch.exp(outputs)
                top_class = probs.max(dim=1)[1]
                equals = top_class == labels
                accuracy += equals.float().mean().item()

        print(f"Test Accuracy: {accuracy / len(self.test_loader):.4f}")

    def save_model(self, path="vgg19_flower_classifier.pth"):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': {i: i for i in range(self.num_classes)},
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path="vgg19_flower_classifier.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}")

    def _plot_metrics(self, train_losses, val_losses, val_accuracies):
        epochs = range(1, self.epochs + 1)

        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss vs. Epochs")
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_metrics_to_csv(self, train_losses, val_losses, val_accuracies, output_csv="metrics.csv"):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])

            for epoch, (train_loss, val_loss, val_accuracy) in enumerate(zip(train_losses, val_losses, val_accuracies),
                                                                         start=1):
                writer.writerow([epoch, train_loss, val_loss, val_accuracy])

        print(f"Metrics saved to {output_csv}")


class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    image_dir = "102flowers/images"
    label_path = "102flowers/imagelabels.mat"
    classifier = FlowerClassifier(image_dir, label_path)

    # load last_check point
    print("Loading latest model...")
    classifier.load_model("vgg19_flower_classifier.pth")

    print("Training the model...")
    classifier.train()

    print("Saving model...")
    classifier.save_model()

    print("Testing the model...")
    classifier.test()

    # plot metrics
    df = pd.read_csv('VGG_metrics.csv')
    classifier._plot_metrics(train_losses=df['Train Loss'], val_losses=df['Validation Loss'], val_accuracies=df['Validation Accuracy'])


