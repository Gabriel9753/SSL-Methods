import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_downloader.dataset_utils import get_dataset_dfs
from torch_utils.custom_dataset import CustomDataset
from torch_utils.models import SimpleNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def confusion_matrix_plot(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = np.zeros((len(np.unique(all_labels)), len(np.unique(all_labels))), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return accuracy, save_path


def evaluate(model, data_loader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def visualize_false_predictions(model, data_loader, device, max_per_class=5):
    """
    Visualize false predictions for each class.
    Saves a figure with up to max_per_class incorrect predictions per class.
    """
    model.eval()

    # Dictionary to store misclassified images by true class
    misclassified = {}
    original_images = {}

    with torch.no_grad():
        for images, labels, image_paths in data_loader:
            original_batch = images.clone()  # Keep the original images before flattening
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Find misclassified images
            mask = (predicted != labels)
            for i in range(len(mask)):
                if mask[i]:
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    if true_label not in misclassified:
                        misclassified[true_label] = []
                        original_images[true_label] = []

                    if len(misclassified[true_label]) < max_per_class:
                        misclassified[true_label].append((true_label, pred_label))
                        original_images[true_label].append(original_batch[i].cpu())

    num_classes = len(misclassified)
    if num_classes == 0:
        print("No misclassified images found!")
        return

    fig, axes = plt.subplots(num_classes, min(max_per_class, 5), figsize=(15, 2*num_classes))
    if num_classes == 1:
        axes = np.array([axes])

    for i, class_idx in enumerate(sorted(misclassified.keys())):
        examples = misclassified[class_idx]
        images = original_images[class_idx]

        for j in range(min(len(examples), max_per_class)):
            if j < len(examples):
                true_label, pred_label = examples[j]
                img = images[j]

                if img.shape[0] == 1:
                    img = img.squeeze(0)
                else:
                    img = img.permute(1, 2, 0)

                img = img * 0.5 + 0.5

                ax = axes[i, j]
                ax.imshow(img, cmap='gray' if img.shape[-1] != 3 else None)
                ax.set_title(f'True: {true_label}\nPred: {pred_label}')
                ax.axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'false_predictions.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path

def main():
    dfs = get_dataset_dfs("data/mnist")
    train_df = dfs["train"]
    test_df = dfs["test"]

    # Define transformations
    transform = [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    train_dataset = CustomDataset(
        image_paths=train_df["image_path"].tolist(),
        labels=train_df["class_id"].tolist(),
        transform=transforms.Compose(transform + [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]),
        preload_images=True,
    )

    test_dataset = CustomDataset(
        image_paths=test_df["image_path"].tolist(),
        labels=test_df["class_id"].tolist(),
        transform=transforms.Compose(transform),
        preload_images=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    simple_nn = SimpleNN(
        input_size=28 * 28,
        num_classes=len(train_df["class_id"].unique()),
        hidden_layers=[512, 256, 128],
        dropout=0.05,
        activation="relu",
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simple_nn.parameters(), lr=0.0008)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_nn.to(device)

    num_epochs = 15
    losses = []
    epoch_losses = []
    accuracies = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as epoch_pbar:
        for epoch in range(num_epochs):
            simple_nn.train()
            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(total=len(train_loader), desc="Batch", unit="batch", leave=False) as batch_pbar:
                for images, labels, _ in train_loader:
                    images = images.view(images.size(0), -1).to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = simple_nn(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    losses.append(loss.item())
                    accuracies.append(100 * correct / total)

                    batch_pbar.update(1)
                    batch_pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            epoch_pbar.update(1)
            epoch_pbar.set_postfix(loss=epoch_loss, accuracy=epoch_accuracy)

            val_loss, val_accuracy = evaluate(simple_nn, test_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

    # print top-5 accuracies and losses
    print(f"Top-5 accuracies: {sorted(epoch_accuracies)[-5:]}")
    print(f"Top-5 losses: {sorted(epoch_losses)[:5]}")
    print(f"Final training loss: {epoch_losses[-1]}")
    print(f"Final training accuracy: {epoch_accuracies[-1]}")

    # top-5 accuracies and losses for validation set
    print(f"Top-5 validation accuracies: {sorted(val_accuracies)[-5:]}")
    print(f"Top-5 validation losses: {sorted(val_losses)[:5]}")
    print(f"Final validation loss: {val_losses[-1]}")
    print(f"Final validation accuracy: {val_accuracies[-1]}")

    # confusion matrix
    accuracy, cm_path = confusion_matrix_plot(simple_nn, test_loader, device)
    print(f"Confusion matrix saved at {cm_path}")

    # Add visualization of false predictions
    false_pred_path = visualize_false_predictions(simple_nn, test_loader, device)
    if false_pred_path:
        print(f"False predictions visualization saved at {false_pred_path}")

    # plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "losses.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracies")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracies.png"), dpi=300)
    plt.close()



if __name__ == "__main__":
    main()