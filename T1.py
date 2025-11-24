import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# PHASE 1: PREPROCESSING
# ============================================================================

class SMDDataset(Dataset):
    """Custom Dataset for SMD data"""
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def load_smd_data(data_path):
    """
    Load SMD dataset from directory structure
    
    Args:
        data_path: Path to data folder containing train/, test/, test_label/
    
    Returns:
        Dictionary with data for each machine
    """
    print("Loading SMD dataset...")
    
    data_dict = {}
    
    # Get list of machines (assuming naming convention like machine-1-1.txt)
    train_files = sorted(os.listdir(os.path.join(data_path, 'train')))
    
    print(f"Found {len(train_files)} machines")
    
    for i, filename in enumerate(train_files):
        machine_id = filename.replace('.txt', '')
        
        # Load train data
        train_file = os.path.join(data_path, 'train', filename)
        train_data = pd.read_csv(train_file, header=None).values
        
        # Load test data
        test_file = os.path.join(data_path, 'test', filename)
        test_data = pd.read_csv(test_file, header=None).values
        
        # Load test labels
        label_file = os.path.join(data_path, 'test_label', filename)
        test_labels = pd.read_csv(label_file, header=None).values.flatten()
        
        data_dict[machine_id] = {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels
        }
        
        print(f"  Machine {i+1}: train {train_data.shape}, test {test_data.shape}")
    
    return data_dict


def normalize_data(train_data, test_data):
    """
    Normalize train and test data using StandardScaler
    
    Args:
        train_data: Training data (n_samples, n_features)
        test_data: Test data (n_samples, n_features)
    
    Returns:
        Normalized train and test data, and the scaler
    """
    scaler = StandardScaler()
    
    # Fit on train data
    train_normalized = scaler.fit_transform(train_data)
    
    # Transform test data using train statistics
    test_normalized = scaler.transform(test_data)
    
    print(f"Data normalized:")
    print(f"  Train - Mean: {train_normalized.mean():.4f}, Std: {train_normalized.std():.4f}")
    print(f"  Test - Mean: {test_normalized.mean():.4f}, Std: {test_normalized.std():.4f}")
    
    return train_normalized, test_normalized, scaler


def create_sequences(data, window_size=100, step=5):
    """
    Create sliding window sequences from time series data
    
    Args:
        data: Input time series (n_samples, n_features)
        window_size: Length of each sequence
        step: Step size for sliding window (default=5 to reduce memory)
    
    Returns:
        Sequences (n_sequences, window_size, n_features)
    """
    sequences = []
    
    for i in range(0, len(data) - window_size + 1, step):
        seq = data[i:i + window_size]
        sequences.append(seq)
    
    return np.array(sequences)


def create_label_sequences(labels, window_size=100, step=5):
    """
    Create sequences of labels (1 if any anomaly in window, 0 otherwise)
    
    Args:
        labels: Binary labels (n_samples,)
        window_size: Length of each sequence
        step: Step size for sliding window
    
    Returns:
        Label sequences (n_sequences,)
    """
    label_seqs = []
    
    for i in range(0, len(labels) - window_size + 1, step):
        label_seq = labels[i:i + window_size]
        # Label is 1 if any anomaly exists in the window
        label_seqs.append(1 if np.any(label_seq == 1) else 0)
    
    return np.array(label_seqs)


def preprocess_machine_data(train_data, test_data, test_labels, window_size=100):
    """
    Complete preprocessing pipeline for one machine
    
    Args:
        train_data: Raw training data
        test_data: Raw test data
        test_labels: Ground truth test labels
        window_size: Sequence length
    
    Returns:
        Preprocessed sequences and labels
    """
    # Normalize
    train_norm, test_norm, scaler = normalize_data(train_data, test_data)
    
    # Create sequences
    X_train = create_sequences(train_norm, window_size=window_size, step=1)
    X_test = create_sequences(test_norm, window_size=window_size, step=1)
    y_test = create_label_sequences(test_labels, window_size=window_size, step=1)
    
    print(f"Sequences created:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Anomaly percentage in test: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_test, scaler


def prepare_all_machines(data_dict, window_size=100):
    """
    Prepare data for all machines (one at a time to save memory)
    
    Args:
        data_dict: Dictionary with data for each machine
        window_size: Sequence length
    
    Returns:
        Dictionary with preprocessed data for each machine
    """
    preprocessed_dict = {}
    
    for idx, (machine_id, data) in enumerate(data_dict.items()):
        print(f"\n[{idx+1}/{len(data_dict)}] Processing {machine_id}...")
        
        try:
            X_train, X_test, y_test, scaler = preprocess_machine_data(
                data['train'],
                data['test'],
                data['test_labels'],
                window_size=window_size
            )
            
            preprocessed_dict[machine_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': scaler
            }
        except MemoryError:
            print(f"  WARNING: Skipped {machine_id} due to memory constraints")
            continue
    
    return preprocessed_dict


def create_dataloaders(X_train, X_test, y_test, batch_size=32, val_split=0.1):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        X_train: Training sequences
        X_test: Test sequences
        y_test: Test labels
        batch_size: Batch size
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split training data into train and validation
    n_train = len(X_train)
    n_val = int(n_train * val_split)
    
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    X_train_split = X_train[train_indices]
    X_val_split = X_train[val_indices]
    
    # Create datasets
    train_dataset = SMDDataset(X_train_split)
    val_dataset = SMDDataset(X_val_split)
    test_dataset = SMDDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def visualize_data_sample(X_train, X_test, n_features_to_show=5):
    """
    Visualize sample of train and test data
    
    Args:
        X_train: Training sequences
        X_test: Test sequences
        n_features_to_show: Number of features to visualize
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot training sample
    sample_train = X_train[0, :, :n_features_to_show]
    for i in range(n_features_to_show):
        axes[0].plot(sample_train[:, i], label=f'Feature {i+1}')
    axes[0].set_title('Training Sample (First Sequence)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Normalized Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot test sample
    sample_test = X_test[0, :, :n_features_to_show]
    for i in range(n_features_to_show):
        axes[1].plot(sample_test[:, i], label=f'Feature {i+1}')
    axes[1].set_title('Test Sample (First Sequence)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Normalized Value')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('data_sample.png', dpi=100, bbox_inches='tight')
    print("Visualization saved as 'data_sample.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'data'  # Path to SMD dataset
    WINDOW_SIZE = 100
    BATCH_SIZE = 32
    
    # Step 1: Load raw data
    print("="*70)
    print("PHASE 1: DATA PREPROCESSING")
    print("="*70)
    
    data_dict = load_smd_data(DATA_PATH)
    
    # Step 2: Preprocess first machine as example
    first_machine = list(data_dict.keys())[0]
    print(f"\n{'='*70}")
    print(f"Preprocessing {first_machine} as example...")
    print(f"{'='*70}")
    
    machine_data = data_dict[first_machine]
    X_train, X_test, y_test, scaler = preprocess_machine_data(
        machine_data['train'],
        machine_data['test'],
        machine_data['test_labels'],
        window_size=WINDOW_SIZE
    )
    
    # Step 3: Create DataLoaders
    print(f"\n{'='*70}")
    print("Creating DataLoaders...")
    print(f"{'='*70}")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_test, y_test, 
        batch_size=BATCH_SIZE
    )
    
    # Step 4: Visualize sample data
    print(f"\n{'='*70}")
    print("Visualizing data samples...")
    print(f"{'='*70}")
    visualize_data_sample(X_train, X_test, n_features_to_show=5)
    
    # Step 5: Preprocess all machines (optional - for full dataset)
    print(f"\n{'='*70}")
    print("Preprocessing all machines (this may take a few minutes)...")
    print(f"{'='*70}")
    preprocessed_dict = prepare_all_machines(data_dict, window_size=WINDOW_SIZE)
    
    print(f"\n{'='*70}")
    print("PHASE 1 COMPLETE!")
    print(f"{'='*70}")
    print(f"Ready for Phase 2: Geometric Masking")
    print(f"Successfully preprocessed: {len(preprocessed_dict)}/{len(data_dict)} machines")
    
    # Save first machine data for next phases
    first_key = list(preprocessed_dict.keys())[0]
    first_data = preprocessed_dict[first_key]
    np.savez('preprocessed_data.npz',
             X_train=first_data['X_train'],
             X_test=first_data['X_test'],
             y_test=first_data['y_test'])
    print(f"Sample preprocessed data saved to 'preprocessed_data.npz' (from {first_key})")
    
    # Save all machines data as dictionary
    import pickle
    with open('all_machines_preprocessed.pkl', 'wb') as f:
        pickle.dump(preprocessed_dict, f)
    print(f"All machines saved to 'all_machines_preprocessed.pkl'")

    