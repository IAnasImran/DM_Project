import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ============================================================================
# PHASE 3: TRANSFORMER ARCHITECTURE + CONTRASTIVE LOSS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for time series"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature extraction"""
    
    def __init__(self, input_dim, hidden_dim=64, n_heads=8, n_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            encoded: (batch_size, seq_len, hidden_dim)
            representation: (batch_size, hidden_dim) - global average pooling
        """
        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        # Global average pooling for representation
        representation = encoded.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Project output
        representation = self.output_projection(representation)
        
        return encoded, representation


class ContrastiveLoss(nn.Module):
    """NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)"""
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature scaling parameter
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Encoded anchor (batch_size, hidden_dim)
            z_j: Encoded positive (batch_size, hidden_dim)
        
        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, hidden_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # (2*batch_size,)
        
        # Compute loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        
        return loss


class TransformerModel(nn.Module):
    """Complete Transformer model with contrastive learning"""
    
    def __init__(self, input_dim, hidden_dim=64, n_heads=8, n_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.encoder = TransformerEncoder(input_dim, hidden_dim, n_heads, n_layers, dropout)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
    
    def forward(self, x1, x2=None):
        """
        Args:
            x1: Anchor sequences (batch_size, seq_len, input_dim)
            x2: Positive sequences (batch_size, seq_len, input_dim), optional
        
        Returns:
            If x2 is None: encoded, representation
            If x2 is provided: loss, encoded1, encoded2, rep1, rep2
        """
        encoded1, rep1 = self.encoder(x1)
        
        if x2 is None:
            return encoded1, rep1
        
        encoded2, rep2 = self.encoder(x2)
        loss = self.contrastive_loss(rep1, rep2)
        
        return loss, encoded1, encoded2, rep1, rep2


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch with contrastive learning"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for anchor, positive in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        
        optimizer.zero_grad()
        
        loss, _, _, _, _ = model(anchor, positive)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for anchor, positive in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            
            loss, _, _, _, _ = model(anchor, positive)
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def train_transformer(model, train_loader, val_loader, epochs=10, lr=1e-3, 
                     device='cuda', checkpoint_path='transformer_best.pt'):
    """
    Train Transformer with contrastive learning
    
    Args:
        model: TransformerModel instance
        train_loader: Training DataLoader with (anchor, positive) pairs
        val_loader: Validation DataLoader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        checkpoint_path: Path to save best model
    
    Returns:
        train_losses, val_losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining Transformer with Contrastive Learning")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved (Val Loss: {val_loss:.6f})")
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"\nBest model loaded from {checkpoint_path}")
    
    return train_losses, val_losses, model


def extract_representations(model, dataloader, device):
    """Extract representations for all samples"""
    model.eval()
    representations = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting representations")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            _, rep = model.encoder(x)
            representations.append(rep.cpu().numpy())
    
    return np.vstack(representations)


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """Plot training history"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Transformer Training History (Contrastive Learning)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PHASE 3: TRANSFORMER ARCHITECTURE + CONTRASTIVE LOSS")
    print("="*70)
    
    # Load data from Phase 2 (load raw numpy arrays instead of dataloaders)
    print("\nLoading data from Phase 2...")
    try:
        # Try to load the npz file directly
        data_npz = np.load('preprocessed_data.npz')
        X_train = data_npz['X_train']
        X_test = data_npz['X_test']
        y_test = data_npz['y_test']
        print(f"Data loaded from preprocessed_data.npz")
    except:
        # Fallback: load from pickle
        import pickle
        with open('all_machines_preprocessed.pkl', 'rb') as f:
            preprocessed_dict = pickle.load(f)
        first_machine = list(preprocessed_dict.keys())[0]
        X_train = preprocessed_dict[first_machine]['X_train']
        X_test = preprocessed_dict[first_machine]['X_test']
        y_test = preprocessed_dict[first_machine]['y_test']
        print(f"Data loaded from all_machines_preprocessed.pkl ({first_machine})")
    
    print(f"Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Configuration
    INPUT_DIM = X_train.shape[2]  # Number of features
    HIDDEN_DIM = 64
    N_HEADS = 8
    N_LAYERS = 2
    DROPOUT = 0.1
    EPOCHS = 5
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nModel Configuration:")
    print(f"  Input Dimension: {INPUT_DIM}")
    print(f"  Hidden Dimension: {HIDDEN_DIM}")
    print(f"  Number of Heads: {N_HEADS}")
    print(f"  Number of Layers: {N_LAYERS}")
    print(f"  Device: {DEVICE}")
    
    # Create contrastive dataloaders from Phase 2 code
    print(f"\n{'='*70}")
    print("Creating Contrastive DataLoaders...")
    print(f"{'='*70}")
    
    # Import augmentation from Phase 2
    from torch.utils.data import Dataset
    
    class GeometricMaskingSimple:
        """Simple version for Phase 3"""
        def __init__(self, mask_ratio=0.2):
            self.mask_ratio = mask_ratio
        
        def augment(self, x):
            x_aug = x.copy()
            n_features = x.shape[1]
            n_mask = max(1, int(n_features * self.mask_ratio))
            mask_idx = np.random.choice(n_features, size=n_mask, replace=False)
            x_aug[:, mask_idx] = 0
            return x_aug
    
    class ContrastiveDatasetSimple(Dataset):
        def __init__(self, X, augmenter):
            self.X = X
            self.augmenter = augmenter
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            anchor = self.X[idx].astype(np.float32)
            positive = self.augmenter.augment(self.X[idx]).astype(np.float32)
            return torch.FloatTensor(anchor), torch.FloatTensor(positive)
    
    augmenter = GeometricMaskingSimple(mask_ratio=0.2)
    contrastive_dataset = ContrastiveDatasetSimple(X_train, augmenter)
    contrastive_train_loader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True)
    
    print(f"Contrastive DataLoader created with {len(contrastive_train_loader)} batches")
    
    # Create model
    print(f"\n{'='*70}")
    print("Creating Transformer Model...")
    print(f"{'='*70}")
    
    model = TransformerModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train transformer
    print(f"\n{'='*70}")
    print("Training Transformer with Contrastive Loss...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    train_losses, val_losses, best_model = train_transformer(
        model,
        contrastive_train_loader,
        contrastive_train_loader,  # Use same data for demo
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
        checkpoint_path='transformer_best.pt'
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training history
    print(f"\n{'='*70}")
    print("Plotting training history...")
    print(f"{'='*70}")
    plot_training_history(train_losses, val_losses)
    
    # Extract representations
    print(f"\n{'='*70}")
    print("Extracting representations...")
    print(f"{'='*70}")
    
    # Create regular dataloaders for representation extraction
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader_extract = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader_extract = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_representations = extract_representations(best_model, train_loader_extract, DEVICE)
    test_representations = extract_representations(best_model, test_loader_extract, DEVICE)
    
    print(f"Train representations shape: {train_representations.shape}")
    print(f"Test representations shape: {test_representations.shape}")
    
    # Save model and representations
    print(f"\n{'='*70}")
    print("Saving model and representations...")
    print(f"{'='*70}")
    
    torch.save({
        'model_state': best_model.state_dict(),
        'train_representations': train_representations,
        'test_representations': test_representations,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS
        }
    }, 'transformer_phase3.pt')
    
    print("Model and representations saved to 'transformer_phase3.pt'")
    
    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE!")
    print(f"{'='*70}")
    print(f"Ready for Phase 4: GAN Training")
    print(f"\nGenerated Files:")
    print(f"  - transformer_best.pt (model weights)")
    print(f"  - transformer_phase3.pt (complete data for Phase 4)")
    print(f"  - training_history.png")