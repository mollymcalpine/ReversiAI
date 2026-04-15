"""
Train the Reversi evaluation network.

Architecture:
A small convolutional neural network that takes a (3, 8, 8) board encoding, and
outputs a scalar in [-1, 1] (representing how good the position is for Black).

Input: (batch, 3, 8, 8)
Conv1: 64 filters, 3x3, padding=1 -> (batch, 64, 8, 8)
Conv2: 128 filters, 3x3, padding=1 -> (batch, 128, 8, 8)
Conv3: 128 filters, 3x3, padding=1 -> (batch, 128, 8, 8)
Flatten: -> (batch, 8192)
FC1: -> (batch, 256)
FC2: -> (batch, 64)
Output: -> (batch, 1) with tanh activation (range -1 to +1)

Run from the ReversiAI root direction:
    python scripts/train.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#-----------------------------------------------------------------------
# Configuration:
#-----------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "reversi_net.pt")

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
VAL_SPLIT = 0.1 # 10% of data used for validation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#-----------------------------------------------------------------------
# Dataset:
#-----------------------------------------------------------------------

class ReversiDataset(Dataset):
    def __init__(self, boards: np.ndarray, outcomes: np.ndarray) -> None:
        self.boards = torch.tensor(boards, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.boards)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.outcomes[idx]
    
#-----------------------------------------------------------------------
# Model:
#-----------------------------------------------------------------------

class ReversiNet(nn.Module):
    """
    Small CNN for Reversi board evalutation.
    Input: (batch, 3, 8, 8)
    Output: (batch, 1) in range [-1, 1]
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1:
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2:
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 3:
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(), # Squash output to [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
    
#-----------------------------------------------------------------------
# Training loop:
#-----------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for boards, outcomes in loader:
        boards, outcomes = boards.to(device), outcomes.to(device)
        optimizer.zero_grad()
        predictions = model(boards)
        loss = criterion(predictions, outcomes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(boards)
    return total_loss / len(loader.dataset)

def val_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for boards, outcomes in loader:
            boards, outcomes = boards.to(device), outcomes.to(device)
            predictions = model(boards)
            total_loss += criterion(predictions, outcomes).item() * len(boards)
    return total_loss / len(loader.dataset)

#-----------------------------------------------------------------------
# Main:
#-----------------------------------------------------------------------

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data:
    boards_path = os.path.join(DATA_DIR, "boards.npy")
    outcomes_path = os.path.join(DATA_DIR, "outcomes.npy")

    if not os.path.exists(boards_path):
        print("ERROR: Training data not found.")
        print("Run 'python scripts/generate_data.py' first.")
        sys.exit(1)

    print("Loading data...")
    boards = np.load(boards_path)
    outcomes = np.load(outcomes_path)
    print(f"  {len(boards):,} positions loaded")

    # Train / val split:
    dataset = ReversiDataset(boards, outcomes)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {train_size:,}  |  Val: {val_size:,}")
    print(f"  Device: {DEVICE}\n")

    # Model, optimizer, loss:
    model = ReversiNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = val_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        marker = "  ✓ best" if val_loss < best_val_loss else ""

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  {current_lr:>10.2e}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val loss : {best_val_loss:.6f}")
    print(f"Model saved : {MODEL_PATH}")

if __name__ == "__main__":
    main()
