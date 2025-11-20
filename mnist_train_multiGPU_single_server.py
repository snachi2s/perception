import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# -------------------------
# Model definition
# -------------------------
class SimpleCNN(nn.Module):
    """A small CNN suitable for MNIST (28x28 grayscale images)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Training / evaluation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


# -------------------------
# Distributed setup helpers
# -------------------------
def setup_distributed():
    """Initialize distributed process group using torchrun environment variables."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# -------------------------
# Main script
# -------------------------
def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 5
    learning_rate = 1e-3

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    if is_main_process(rank):
        print(f"World size: {world_size}, using device: {device}")

    # TensorBoard: only on main process to avoid duplicate logs
    if is_main_process(rank):
        log_dir = os.path.join(
            "runs",
            f"mnist_cnn_ddp_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=is_main_process(rank),  # only main rank downloads
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=False,  # already downloaded by main rank
        transform=transform,
    )

    # Ensure all ranks wait until data is ready
    dist.barrier()

    # Use DistributedSampler so each GPU sees different data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = SimpleCNN(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: log graph from main process
    if is_main_process(rank):
        example_images, _ = next(iter(train_loader))
        writer.add_graph(model, example_images.to(device))

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Important for shuffling with DistributedSampler
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        if is_main_process(rank):
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

    if is_main_process(rank):
        os.makedirs("checkpoints", exist_ok=True)
        model_path = os.path.join("checkpoints", "mnist_cnn_ddp.pt")
        # model.module holds the underlying model when using DDP
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            model_path,
        )
        print(f"Model saved to {model_path}")
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
