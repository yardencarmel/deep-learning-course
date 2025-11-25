import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)


class LeNet5(nn.Module):
    """
    LeNet-5 architecture adapted for Fashion-MNIST (28x28 images)
    Original LeNet-5 was designed for 32x32 images, this is the MNIST variant
    """
    def __init__(self, use_dropout=False, use_batch_norm=False):
        super(LeNet5, self).__init__()
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 28x28 -> 24x24 -> 12x12 (after pool)
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)
            self.bn3 = nn.BatchNorm1d(120)
            self.bn4 = nn.BatchNorm1d(84)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 12x12 -> 5x5 after second pool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for Fashion-MNIST
        
        # Dropout layer
        if use_dropout:
            self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def forward(self, x, training=True):
        # First conv block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second conv block
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 12x12 -> 6x6 -> 5x5 (with padding considerations)
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # First FC layer
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        if self.use_dropout and training:
            x = self.dropout(x)
        
        # Second FC layer
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = F.relu(x)
        if self.use_dropout and training:
            x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x


def get_data_loaders():
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, use_weight_decay=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, training=True)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, training_mode=False):
    """Evaluate model on dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            # For dropout: training_mode=False means dropout is disabled
            outputs = model(images, training=training_mode)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train_model(config_name, use_dropout=False, use_batch_norm=False, use_weight_decay=False):
    """Train model with specified configuration"""
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"Dropout: {use_dropout}, BatchNorm: {use_batch_norm}, WeightDecay: {use_weight_decay}")
    print(f"{'='*60}\n")
    
    # Create model
    model = LeNet5(use_dropout=use_dropout, use_batch_norm=use_batch_norm).to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Use weight_decay in optimizer if specified
    if use_weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Get data
    train_loader, test_loader = get_data_loaders()
    
    # Training history
    train_accs = []  # Training accuracy (with dropout if applicable)
    train_eval_accs = []  # Training accuracy without dropout (for dropout models)
    test_accs = []
    
    best_test_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        train_accs.append(train_acc)
        
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, training_mode=False)
        test_accs.append(test_acc)
        
        # For dropout models, also evaluate train set without dropout
        if use_dropout:
            train_eval_acc = evaluate(model, train_loader, training_mode=False)
            train_eval_accs.append(train_eval_acc)
            print(f'Train Loss: {train_loss:.4f}, Train Acc (with dropout): {train_acc:.2f}%, '
                  f'Train Acc (no dropout): {train_eval_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        else:
            train_eval_accs.append(train_acc)  # Same as train_acc for non-dropout models
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'models/{config_name}_best.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'models/{config_name}_final.pth')
    
    # For dropout models, return train_eval_accs (without dropout) for plotting
    if use_dropout:
        return train_eval_accs, test_accs, model
    else:
        return train_accs, test_accs, model


def plot_convergence(config_name, train_accs, test_accs):
    """Plot convergence graphs"""
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Convergence Graph: {config_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(epochs))
    plt.ylim(80, 100)
    plt.tight_layout()
    plt.savefig(f'plots/{config_name}_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: plots/{config_name}_convergence.png")


def create_summary_table(results):
    """Create and save summary table"""
    print("\n" + "="*80)
    print("FINAL ACCURACIES SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Train Accuracy':<20} {'Test Accuracy':<20}")
    print("-"*80)
    
    for config_name, train_acc, test_acc in results:
        print(f"{config_name:<30} {train_acc:>18.2f}% {test_acc:>18.2f}%")
    
    print("="*80)
    
    # Save to file
    with open('results_summary.txt', 'w') as f:
        f.write("FINAL ACCURACIES SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Configuration':<30} {'Train Accuracy':<20} {'Test Accuracy':<20}\n")
        f.write("-"*80 + "\n")
        for config_name, train_acc, test_acc in results:
            f.write(f"{config_name:<30} {train_acc:>18.2f}% {test_acc:>18.2f}%\n")
        f.write("="*80 + "\n")
    
    print("\nSummary saved to results_summary.txt")


def main():
    """Main training function"""
    configurations = [
        ('Baseline', False, False, False),
        ('Dropout', True, False, False),
        ('Weight_Decay', False, False, True),
        ('Batch_Normalization', False, True, False),
    ]
    
    all_results = []
    
    for config_name, use_dropout, use_batch_norm, use_weight_decay in configurations:
        train_accs, test_accs, model = train_model(
            config_name, use_dropout, use_batch_norm, use_weight_decay
        )
        
        # Plot convergence
        plot_convergence(config_name, train_accs, test_accs)
        
        # Store results (use final accuracies)
        all_results.append((config_name, train_accs[-1], test_accs[-1]))
    
    # Create summary table
    create_summary_table(all_results)
    
    print("\nTraining completed! Check 'plots/' for convergence graphs and 'models/' for saved weights.")


if __name__ == '__main__':
    main()
