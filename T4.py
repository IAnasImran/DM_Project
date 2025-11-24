import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ============================================================================
# PHASE 4: GAN (GENERATIVE ADVERSARIAL NETWORK) TRAINING
# ============================================================================

class Generator(nn.Module):
    """Generator: learns to generate realistic normal sequences"""
    
    def __init__(self, latent_dim=20, seq_len=100, n_features=38, hidden_dim=128):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            seq_len: Length of sequence
            n_features: Number of features
            hidden_dim: Hidden dimension
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Fully connected layer to expand latent vector to (seq_len, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * seq_len),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * seq_len),
            nn.Linear(hidden_dim * seq_len, hidden_dim * seq_len),
            nn.ReLU()
        )
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer to generate features
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def forward(self, z):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)
        
        Returns:
            Generated sequence (batch_size, seq_len, n_features)
        """
        batch_size = z.shape[0]
        
        # Expand latent vector
        x = self.fc(z)  # (batch_size, hidden_dim * seq_len)
        
        # Reshape to sequence
        x = x.view(batch_size, self.seq_len, self.hidden_dim)
        
        # Generate sequence with LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        # Output layer
        output = self.output_layer(lstm_out)  # (batch_size, seq_len, n_features)
        output = torch.tanh(output)  # Output in [-1, 1]
        
        return output


class Discriminator(nn.Module):
    """Discriminator: distinguishes real vs fake sequences"""
    
    def __init__(self, seq_len=100, n_features=38, hidden_dim=128):
        """
        Args:
            seq_len: Length of sequence
            n_features: Number of features
            hidden_dim: Hidden dimension
        """
        super(Discriminator, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Sequence (batch_size, seq_len, n_features)
        
        Returns:
            Probability (batch_size, 1)
        """
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(x)  # hidden: (num_layers, batch_size, hidden_dim)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Classify
        output = self.fc(last_hidden)  # (batch_size, 1)
        
        return output


class GANModel(nn.Module):
    """Complete GAN model combining Generator and Discriminator"""
    
    def __init__(self, latent_dim=20, seq_len=100, n_features=38, hidden_dim=128):
        super(GANModel, self).__init__()
        
        self.generator = Generator(latent_dim, seq_len, n_features, hidden_dim)
        self.discriminator = Discriminator(seq_len, n_features, hidden_dim)
        self.latent_dim = latent_dim
    
    def generate(self, batch_size, device):
        """Generate fake sequences"""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        fake_seq = self.generator(z)
        return fake_seq
    
    def discriminate(self, x):
        """Discriminate real vs fake"""
        return self.discriminator(x)


def train_gan_epoch(gan, real_loader, device, lr_g=1e-4, lr_d=1e-4):
    """Train GAN for one epoch"""
    
    # Optimizers
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    gan.train()
    total_g_loss = 0
    total_d_loss = 0
    n_batches = 0
    
    pbar = tqdm(real_loader, desc="GAN Training")
    for real_seq in pbar:
        if isinstance(real_seq, (list, tuple)):
            real_seq = real_seq[0]
        
        real_seq = real_seq.to(device)
        batch_size = real_seq.shape[0]
        
        # Labels
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)
        
        # ==================== Train Discriminator ====================
        optimizer_d.zero_grad()
        
        # Real sequences
        d_real_output = gan.discriminator(real_seq)
        loss_d_real = criterion(d_real_output, real_label)
        
        # Fake sequences
        fake_seq = gan.generate(batch_size, device)
        d_fake_output = gan.discriminator(fake_seq.detach())
        loss_d_fake = criterion(d_fake_output, fake_label)
        
        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()
        
        # ==================== Train Generator ====================
        optimizer_g.zero_grad()
        
        # Generate fake sequences
        fake_seq = gan.generate(batch_size, device)
        d_fake_output = gan.discriminator(fake_seq)
        
        # Generator loss (fool discriminator)
        loss_g = criterion(d_fake_output, real_label)
        loss_g.backward()
        optimizer_g.step()
        
        # Track losses
        total_d_loss += loss_d.item()
        total_g_loss += loss_g.item()
        n_batches += 1
        
        pbar.set_postfix({
            'D_loss': loss_d.item(),
            'G_loss': loss_g.item()
        })
    
    return total_d_loss / n_batches, total_g_loss / n_batches


def train_gan(gan, real_loader, epochs=10, device='cuda', checkpoint_path='gan_best.pt'):
    """
    Train GAN
    
    Args:
        gan: GANModel instance
        real_loader: DataLoader with real sequences
        epochs: Number of epochs
        device: Device to use
        checkpoint_path: Path to save best model
    
    Returns:
        d_losses, g_losses, trained_gan
    """
    gan = gan.to(device)
    
    d_losses = []
    g_losses = []
    best_g_loss = float('inf')
    
    print(f"\nTraining GAN")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        d_loss, g_loss = train_gan_epoch(gan, real_loader, device, lr_g=1e-4, lr_d=1e-4)
        
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        print(f"Discriminator Loss: {d_loss:.6f} | Generator Loss: {g_loss:.6f}")
        
        # Save best model (best generator)
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save(gan.state_dict(), checkpoint_path)
            print(f"Best model saved (G_loss: {g_loss:.6f})")
    
    # Load best model
    gan.load_state_dict(torch.load(checkpoint_path))
    print(f"\nBest model loaded from {checkpoint_path}")
    
    return d_losses, g_losses, gan


def generate_samples(gan, n_samples, batch_size=32, device='cuda'):
    """Generate fake sequences"""
    gan.eval()
    all_fake = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            fake_seq = gan.generate(current_batch_size, device)
            all_fake.append(fake_seq.cpu().numpy())
    
    return np.vstack(all_fake)


def plot_gan_training_history(d_losses, g_losses, save_path='gan_training_history.png'):
    """Plot GAN training history"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(d_losses) + 1)
    ax.plot(epochs, d_losses, 'b-', label='Discriminator Loss', marker='o', markersize=4)
    ax.plot(epochs, g_losses, 'r-', label='Generator Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('GAN Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"GAN training history saved to {save_path}")
    plt.close()


def plot_generated_vs_real(real_samples, fake_samples, n_samples=5, save_path='generated_vs_real.png'):
    """Compare generated vs real sequences"""
    n_features_show = 5
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
    
    for i in range(n_samples):
        # Real
        real = real_samples[i, :, :n_features_show]
        axes[i, 0].plot(real)
        axes[i, 0].set_title(f'Real Sequence {i+1}', fontsize=11, fontweight='bold')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True)
        axes[i, 0].legend([f'F{j+1}' for j in range(n_features_show)], loc='upper right', fontsize=8)
        
        # Fake
        fake = fake_samples[i, :, :n_features_show]
        axes[i, 1].plot(fake)
        axes[i, 1].set_title(f'Generated Sequence {i+1}', fontsize=11, fontweight='bold')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True)
        axes[i, 1].legend([f'F{j+1}' for j in range(n_features_show)], loc='upper right', fontsize=8)
    
    axes[-1, 0].set_xlabel('Time Steps')
    axes[-1, 1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Generated vs real comparison saved to {save_path}")
    plt.close()


def denormalize_sequences(fake_samples, scaler):
    """Denormalize generated sequences back to original scale"""
    n_samples, seq_len, n_features = fake_samples.shape
    fake_flat = fake_samples.reshape(n_samples * seq_len, n_features)
    fake_denorm = scaler.inverse_transform(fake_flat)
    fake_denorm = fake_denorm.reshape(n_samples, seq_len, n_features)
    return fake_denorm


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PHASE 4: GAN TRAINING")
    print("="*70)
    
    # Load data from Phase 3
    print("\nLoading data from Phase 3...")
    data = torch.load('transformer_phase3.pt', weights_only=False)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']
    config = data['config']
    
    print(f"Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Configuration
    LATENT_DIM = 20
    SEQ_LEN = X_train.shape[1]
    N_FEATURES = X_train.shape[2]
    HIDDEN_DIM = 128
    EPOCHS = 10
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nGAN Configuration:")
    print(f"  Latent Dimension: {LATENT_DIM}")
    print(f"  Sequence Length: {SEQ_LEN}")
    print(f"  Number of Features: {N_FEATURES}")
    print(f"  Hidden Dimension: {HIDDEN_DIM}")
    print(f"  Device: {DEVICE}")
    
    # Create DataLoader
    print(f"\n{'='*70}")
    print("Creating DataLoader...")
    print(f"{'='*70}")
    
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"DataLoader created with {len(train_loader)} batches")
    
    # Create GAN model
    print(f"\n{'='*70}")
    print("Creating GAN Model...")
    print(f"{'='*70}")
    
    gan = GANModel(
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        hidden_dim=HIDDEN_DIM
    )
    
    print(f"GAN created successfully!")
    total_params = sum(p.numel() for p in gan.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  Generator: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
    
    # Train GAN
    print(f"\n{'='*70}")
    print("Training GAN...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    d_losses, g_losses, trained_gan = train_gan(
        gan,
        train_loader,
        epochs=EPOCHS,
        device=DEVICE,
        checkpoint_path='gan_best.pt'
    )
    
    training_time = time.time() - start_time
    print(f"\nGAN training completed in {training_time:.2f} seconds")
    
    # Plot training history
    print(f"\n{'='*70}")
    print("Plotting training history...")
    print(f"{'='*70}")
    plot_gan_training_history(d_losses, g_losses)
    
    # Generate samples
    print(f"\n{'='*70}")
    print("Generating fake sequences...")
    print(f"{'='*70}")
    
    fake_samples = generate_samples(trained_gan, n_samples=len(X_train), 
                                   batch_size=BATCH_SIZE, device=DEVICE)
    
    print(f"Generated samples shape: {fake_samples.shape}")
    
    # Denormalize to original scale for visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, N_FEATURES))
    fake_samples_denorm = denormalize_sequences(fake_samples, scaler)
    real_samples_denorm = denormalize_sequences(X_train, scaler)
    
    # Plot comparison
    print(f"\n{'='*70}")
    print("Plotting generated vs real sequences...")
    print(f"{'='*70}")
    plot_generated_vs_real(real_samples_denorm, fake_samples_denorm, n_samples=5)
    
    # Save everything for Phase 5
    print(f"\n{'='*70}")
    print("Saving GAN model and data...")
    print(f"{'='*70}")
    
    torch.save({
        'gan_state': trained_gan.state_dict(),
        'fake_samples': fake_samples,
        'd_losses': d_losses,
        'g_losses': g_losses,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'config': {
            'latent_dim': LATENT_DIM,
            'seq_len': SEQ_LEN,
            'n_features': N_FEATURES,
            'hidden_dim': HIDDEN_DIM
        }
    }, 'gan_phase4.pt')
    
    print("GAN model and data saved to 'gan_phase4.pt'")
    
    print(f"\n{'='*70}")
    print("PHASE 4 COMPLETE!")
    print(f"{'='*70}")
    print(f"Ready for Phase 5: Testing & Evaluation")
    print(f"\nGenerated Files:")
    print(f"  - gan_best.pt (model weights)")
    print(f"  - gan_phase4.pt (complete data for Phase 5)")
    print(f"  - gan_training_history.png")
    print(f"  - generated_vs_real.png")