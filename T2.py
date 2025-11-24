import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# PHASE 2: GEOMETRIC MASKING & DATA AUGMENTATION
# ============================================================================

class GeometricMasking:
    """Geometric masking techniques for data augmentation"""
    
    def __init__(self, mask_ratio=0.2, noise_std=0.1):
        """
        Args:
            mask_ratio: Ratio of features to mask (0.2 = 20%)
            noise_std: Standard deviation of Gaussian noise
        """
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
    
    def random_feature_masking(self, x):
        """
        Randomly mask features (set to 0)
        
        Args:
            x: Input sequence (seq_len, n_features)
        
        Returns:
            Masked sequence
        """
        x_masked = x.copy()
        n_features = x.shape[1]
        n_mask = max(1, int(n_features * self.mask_ratio))
        
        # Randomly select features to mask
        mask_indices = np.random.choice(n_features, size=n_mask, replace=False)
        x_masked[:, mask_indices] = 0
        
        return x_masked
    
    def temporal_masking(self, x, temporal_ratio=0.1):
        """
        Randomly mask time steps
        
        Args:
            x: Input sequence (seq_len, n_features)
            temporal_ratio: Ratio of time steps to mask
        
        Returns:
            Masked sequence
        """
        x_masked = x.copy()
        seq_len = x.shape[0]
        n_mask = max(1, int(seq_len * temporal_ratio))
        
        # Randomly select time steps to mask
        mask_indices = np.random.choice(seq_len, size=n_mask, replace=False)
        x_masked[mask_indices, :] = 0
        
        return x_masked
    
    def noise_injection(self, x):
        """
        Add Gaussian noise to data
        
        Args:
            x: Input sequence (seq_len, n_features)
        
        Returns:
            Noisy sequence
        """
        noise = np.random.normal(0, self.noise_std, x.shape)
        x_noisy = x + noise
        
        return x_noisy
    
    def scaling_perturbation(self, x, scale_range=(0.8, 1.2)):
        """
        Scale features randomly
        
        Args:
            x: Input sequence (seq_len, n_features)
            scale_range: Tuple of (min_scale, max_scale)
        
        Returns:
            Scaled sequence
        """
        x_scaled = x.copy()
        n_features = x.shape[1]
        
        # Random scale for each feature
        scales = np.random.uniform(scale_range[0], scale_range[1], n_features)
        x_scaled = x_scaled * scales[np.newaxis, :]
        
        return x_scaled
    
    def combined_augmentation(self, x):
        """
        Apply combination of augmentation techniques
        
        Args:
            x: Input sequence (seq_len, n_features)
        
        Returns:
            Augmented sequence
        """
        # Apply 2-3 random augmentations
        augmentations = [
            self.random_feature_masking,
            self.temporal_masking,
            self.noise_injection,
            self.scaling_perturbation
        ]
        
        # Randomly select 2-3 augmentations
        n_aug = np.random.randint(2, 4)
        selected_aug = np.random.choice(len(augmentations), size=n_aug, replace=False)
        
        x_aug = x.copy()
        for idx in selected_aug:
            x_aug = augmentations[idx](x_aug)
        
        return x_aug
    
    def augment(self, x, aug_type='combined'):
        """
        Main augmentation function
        
        Args:
            x: Input sequence
            aug_type: Type of augmentation ('masking', 'temporal', 'noise', 'scaling', 'combined')
        
        Returns:
            Augmented sequence
        """
        if aug_type == 'masking':
            return self.random_feature_masking(x)
        elif aug_type == 'temporal':
            return self.temporal_masking(x)
        elif aug_type == 'noise':
            return self.noise_injection(x)
        elif aug_type == 'scaling':
            return self.scaling_perturbation(x)
        elif aug_type == 'combined':
            return self.combined_augmentation(x)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")


class ContrastiveDataset(Dataset):
    """
    Dataset that returns positive pairs for contrastive learning
    Each sample returns (anchor, positive_augmented)
    """
    def __init__(self, X, augmenter, aug_type='combined'):
        """
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            augmenter: GeometricMasking instance
            aug_type: Type of augmentation to apply
        """
        self.X = X
        self.augmenter = augmenter
        self.aug_type = aug_type
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get anchor (original)
        anchor = self.X[idx].astype(np.float32)
        
        # Create positive (augmented version of same sample)
        positive = self.augmenter.augment(self.X[idx], aug_type=self.aug_type)
        positive = positive.astype(np.float32)
        
        # Convert to tensors
        anchor = torch.FloatTensor(anchor)
        positive = torch.FloatTensor(positive)
        
        return anchor, positive


class RegularDataset(Dataset):
    """Regular dataset for standard training"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def create_augmented_dataset(X_train, batch_size=32, aug_type='combined', 
                           mask_ratio=0.2, noise_std=0.1):
    """
    Create DataLoader with augmented pairs for contrastive learning
    
    Args:
        X_train: Training sequences (n_samples, seq_len, n_features)
        batch_size: Batch size
        aug_type: Type of augmentation
        mask_ratio: Masking ratio
        noise_std: Noise standard deviation
    
    Returns:
        DataLoader returning (anchor, positive) pairs
    """
    augmenter = GeometricMasking(mask_ratio=mask_ratio, noise_std=noise_std)
    dataset = ContrastiveDataset(X_train, augmenter, aug_type=aug_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, augmenter


def create_regular_dataloader(X, y=None, batch_size=32, shuffle=True):
    """
    Create regular DataLoader
    
    Args:
        X: Data sequences
        y: Labels (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    dataset = RegularDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def visualize_augmentations(X_sample, n_augmentations=4):
    """
    Visualize original data and augmented versions
    
    Args:
        X_sample: Single sample (seq_len, n_features)
        n_augmentations: Number of augmentations to show
    """
    augmenter = GeometricMasking(mask_ratio=0.2, noise_std=0.1)
    
    # Select only first 5 features for visualization
    n_features_show = min(5, X_sample.shape[1])
    
    fig, axes = plt.subplots(n_augmentations + 1, 1, figsize=(12, 3*(n_augmentations+1)))
    
    # Plot original
    axes[0].plot(X_sample[:, :n_features_show])
    axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    axes[0].legend([f'Feature {i+1}' for i in range(n_features_show)], loc='upper right')
    
    # Plot augmented versions
    aug_types = ['masking', 'temporal', 'noise', 'scaling']
    
    for i in range(n_augmentations):
        aug_type = aug_types[i % len(aug_types)]
        X_aug = augmenter.augment(X_sample, aug_type=aug_type)
        
        axes[i+1].plot(X_aug[:, :n_features_show])
        axes[i+1].set_title(f'Augmentation: {aug_type.capitalize()}', fontsize=12, fontweight='bold')
        axes[i+1].set_ylabel('Value')
        axes[i+1].grid(True)
        axes[i+1].legend([f'Feature {j+1}' for j in range(n_features_show)], loc='upper right')
    
    axes[-1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig('augmentations_visualization.png', dpi=100, bbox_inches='tight')
    print("Augmentation visualization saved as 'augmentations_visualization.png'")
    plt.close()


def visualize_contrastive_pairs(X_train, n_pairs=3):
    """
    Visualize contrastive pairs (anchor and positive)
    
    Args:
        X_train: Training sequences
        n_pairs: Number of pairs to visualize
    """
    augmenter = GeometricMasking(mask_ratio=0.2, noise_std=0.1)
    n_features_show = min(5, X_train.shape[2])
    
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    
    for i in range(n_pairs):
        idx = np.random.randint(len(X_train))
        anchor = X_train[idx]
        positive = augmenter.augment(X_train[idx], aug_type='combined')
        
        # Plot anchor
        axes[i, 0].plot(anchor[:, :n_features_show])
        axes[i, 0].set_title(f'Anchor (Sample {idx+1})', fontsize=11, fontweight='bold')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True)
        axes[i, 0].legend([f'F{j+1}' for j in range(n_features_show)], loc='upper right')
        
        # Plot positive
        axes[i, 1].plot(positive[:, :n_features_show])
        axes[i, 1].set_title(f'Positive (Augmented)', fontsize=11, fontweight='bold')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True)
        axes[i, 1].legend([f'F{j+1}' for j in range(n_features_show)], loc='upper right')
    
    axes[-1, 0].set_xlabel('Time Steps')
    axes[-1, 1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig('contrastive_pairs_visualization.png', dpi=100, bbox_inches='tight')
    print("Contrastive pairs visualization saved as 'contrastive_pairs_visualization.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import pickle
    
    print("="*70)
    print("PHASE 2: GEOMETRIC MASKING & DATA AUGMENTATION")
    print("="*70)
    
    # Load preprocessed data from Phase 1
    print("\nLoading preprocessed data...")
    try:
        with open('all_machines_preprocessed.pkl', 'rb') as f:
            preprocessed_dict = pickle.load(f)
        print(f"Loaded data for {len(preprocessed_dict)} machines")
    except FileNotFoundError:
        # Fall back to npz file
        data = np.load('preprocessed_data.npz')
        X_train = data['X_train']
        X_test = data['X_test']
        y_test = data['y_test']
        print("Loaded single machine data from preprocessed_data.npz")
    else:
        # Get first machine
        first_machine = list(preprocessed_dict.keys())[0]
        X_train = preprocessed_dict[first_machine]['X_train']
        X_test = preprocessed_dict[first_machine]['X_test']
        y_test = preprocessed_dict[first_machine]['y_test']
        print(f"Using data from: {first_machine}")
    
    # Configuration
    BATCH_SIZE = 32
    MASK_RATIO = 0.2
    NOISE_STD = 0.1
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Step 1: Visualize augmentations
    print(f"\n{'='*70}")
    print("Step 1: Visualizing Augmentation Techniques")
    print(f"{'='*70}")
    sample = X_train[0]
    visualize_augmentations(sample, n_augmentations=4)
    
    # Step 2: Visualize contrastive pairs
    print(f"\n{'='*70}")
    print("Step 2: Visualizing Contrastive Pairs")
    print(f"{'='*70}")
    visualize_contrastive_pairs(X_train, n_pairs=3)
    
    # Step 3: Create contrastive DataLoader
    print(f"\n{'='*70}")
    print("Step 3: Creating Contrastive DataLoaders")
    print(f"{'='*70}")
    
    contrastive_loader, augmenter = create_augmented_dataset(
        X_train,
        batch_size=BATCH_SIZE,
        aug_type='combined',
        mask_ratio=MASK_RATIO,
        noise_std=NOISE_STD
    )
    
    print(f"Contrastive DataLoader created:")
    print(f"  Total batches: {len(contrastive_loader)}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Test one batch
    anchor_batch, positive_batch = next(iter(contrastive_loader))
    print(f"  Anchor batch shape: {anchor_batch.shape}")
    print(f"  Positive batch shape: {positive_batch.shape}")
    
    # Step 4: Create regular DataLoaders for reference
    print(f"\n{'='*70}")
    print("Step 4: Creating Regular DataLoaders (for Phase 3)")
    print(f"{'='*70}")
    
    train_loader = create_regular_dataloader(X_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = create_regular_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train DataLoader:")
    print(f"  Total batches: {len(train_loader)}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    print(f"Test DataLoader:")
    print(f"  Total batches: {len(test_loader)}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Save dataloaders for Phase 3
    print(f"\n{'='*70}")
    print("Step 5: Saving DataLoaders and Augmenter")
    print(f"{'='*70}")
    
    torch.save({
        'contrastive_loader': contrastive_loader,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'augmenter': augmenter,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test
    }, 'dataloaders_phase2.pt')
    
    print("DataLoaders saved to 'dataloaders_phase2.pt'")
    
    print(f"\n{'='*70}")
    print("PHASE 2 COMPLETE!")
    print(f"{'='*70}")
    print(f"Ready for Phase 3: Transformer Architecture")
    print(f"\nGenerated Files:")
    print(f"  - augmentations_visualization.png")
    print(f"  - contrastive_pairs_visualization.png")
    print(f"  - dataloaders_phase2.pt")