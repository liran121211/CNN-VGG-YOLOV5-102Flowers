import os
import csv
import pandas as pd
import scipy.io
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


class FlowerClassifier:
    def __init__(self, image_dir, label_path, num_classes=102, lr=0.001, batch_size=32, epochs=10):
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

        # Create train, validation, and test datasets with respective transforms
        total_size = len(image_paths)
        train_size = int(0.5 * total_size)
        val_size = int(0.25 * total_size)
        test_size = total_size - train_size - val_size

        # Splitting the dataset into subsets
        dataset = FlowerDataset(image_paths, labels)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size , test_size])

        # Apply respective transforms
        train_dataset.dataset = FlowerDataset(image_paths=train_dataset.dataset.image_paths, labels=train_dataset.dataset.labels, transform=self.train_transforms)
        val_dataset.dataset = FlowerDataset(image_paths=val_dataset.dataset.image_paths, labels=val_dataset.dataset.labels, transform=self.val_test_transforms)
        test_dataset.dataset = FlowerDataset(image_paths=test_dataset.dataset.image_paths, labels=test_dataset.dataset.labels, transform=self.val_test_transforms)

        # Creating DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _load_vgg_model(self):
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
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
        test_losses = []
        val_accuracies = []
        test_accuracies = []

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
            val_loss, val_accuracy = self._epoch_validate(criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Test step
            test_loss, test_accuracy = self._epoch_test(criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch + 1}/{self.epochs}], "
                  f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # After training, save the training data
        self.save_metrics_to_csv(train_losses, val_losses, val_accuracies, test_losses, test_accuracies)

    def _epoch_validate(self, criterion):
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

    def _epoch_test(self, criterion):
        self.model.eval()
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.forward(inputs)
                test_loss += criterion(outputs, labels).item()
                probs = torch.exp(outputs)
                top_class = probs.max(dim=1)[1]
                equals = top_class == labels
                accuracy += equals.float().mean().item()

        return test_loss / len(self.test_loader), accuracy / len(self.test_loader)

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
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.class_to_idx = checkpoint['class_to_idx']
            self.model = self.model.to(self.device)
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Model not loaded found in: {path}")
            return

    def _plot_metrics(self, train_losses, val_losses, val_accuracies, test_losses, test_accuracies, df_size):
        epochs = range(1, df_size + 1)

        # Plot Loss
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
        plt.plot(epochs, test_losses, label="Test Loss", linestyle='dotted', marker='x')
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss vs. Epochs", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(range(0, df_size + 1, max(1, df_size // 10)))
        plt.yticks(fontsize=10)

        # Annotate minimum loss for better visualization
        min_val_loss_epoch = val_losses.idxmin() + 1  # idxmin() gets the index of the minimum value
        min_val_loss = val_losses[min_val_loss_epoch - 1]
        plt.annotate(f'Min Val Loss\n{min_val_loss:.2f} (Epoch {min_val_loss_epoch})',
                     xy=(min_val_loss_epoch, min_val_loss),
                     xytext=(min_val_loss_epoch -5, min_val_loss + 0.5),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=10)

        min_test_loss_epoch = test_losses.idxmin() + 1  # idxmin() gets the index of the minimum value
        min_test_loss = test_losses[min_test_loss_epoch - 1]
        plt.annotate(f'Min Test Loss\n{min_test_loss:.2f} (Epoch {min_test_loss_epoch})',
                     xy=(min_test_loss_epoch, min_test_loss),
                     xytext=(min_test_loss_epoch -2, min_test_loss + 0.3),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=10)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green', marker='o')
        plt.plot(epochs, test_accuracies, label="Test Accuracy", color='red', linestyle='dotted', marker='x')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Epochs", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(range(0, df_size + 1, max(1, df_size // 10)))
        plt.yticks(fontsize=10)

        # If `val_accuracies` is a Pandas Series
        max_val_accuracy_epoch = val_accuracies.idxmax() + 1  # Get the index of the maximum value
        max_val_accuracy = val_accuracies[max_val_accuracy_epoch - 1]

        # Annotate the plot
        plt.annotate(f'Max Val Accuracy\n{max_val_accuracy:.2%} (Epoch {max_val_accuracy_epoch})',
                     xy=(max_val_accuracy_epoch, max_val_accuracy),
                     xytext=(max_val_accuracy_epoch - 5, max_val_accuracy - 0.1),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=10)

        # If `val_accuracies` is a Pandas Series
        max_test_accuracy_epoch = test_accuracies.idxmax() + 1  # Get the index of the maximum value
        max_test_accuracy = test_accuracies[max_test_accuracy_epoch - 1]

        # Annotate the plot
        plt.annotate(f'Max Test Accuracy\n{max_test_accuracy:.2%} (Epoch {max_test_accuracy_epoch})',
                     xy=(max_test_accuracy_epoch, max_test_accuracy),
                     xytext=(max_test_accuracy_epoch - 10, max_test_accuracy - 0.2),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=10)

        plt.tight_layout()
        plt.show()

    def save_metrics_to_csv(self, train_losses, val_losses, val_accuracies, test_losses, test_accuracies, output_csv="VGG_metrics.csv"):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"])

            for epoch, (train_loss, val_loss, val_accuracy, test_loss, test_accuracy) in enumerate(
                    zip(train_losses, val_losses, val_accuracies, test_losses, test_accuracies), start=1):
                writer.writerow([epoch, train_loss, val_loss, val_accuracy, test_loss, test_accuracy])

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
    classifier._plot_metrics(
        train_losses=df['Train Loss'],
        val_losses=df['Validation Loss'],
        val_accuracies=df['Validation Accuracy'],
        test_losses=df['Test Loss'],
        test_accuracies=df['Test Accuracy'],
        df_size=len(df)
    )


