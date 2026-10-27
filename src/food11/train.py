"""Train a resnet18 on Food-11 and track the run with mlflow."""

import argparse
from pathlib import Path

import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import mlflow

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASETS = {"processed": "food11_processed", "mini": "food11_processed_mini"}
NUM_CLASSES = 11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def make_loader(root, split, batch_size, shuffle):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root / split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    args = parse_args()
    root = DATA_DIR / DATASETS[args.dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(root, "training", args.batch_size, shuffle=True)
    val_loader = make_loader(root, "validation", args.batch_size, shuffle=False)
    test_loader = make_loader(root, "evaluation", args.batch_size, shuffle=False)

    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    with mlflow.start_run():
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
        })

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")

        _, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"test_accuracy={test_accuracy:.4f}")

        example = next(iter(test_loader))[0][:1].numpy()
        mlflow.pytorch.log_model(model.cpu(), "model", input_example=example)


if __name__ == "__main__":
    main()
