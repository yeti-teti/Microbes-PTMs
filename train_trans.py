import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print, progress
import math

# -----------------------
#     Dataset Class
# -----------------------
class PeptideDataset(Dataset):
    def __init__(self, features, targets):
        """
        features: numpy array of shape (n_samples, n_features)
        targets: numpy array of shape (n_samples,)
        """
        # Group features by property type
        self.basicity = torch.FloatTensor(features[:, :20])       # First 20 features are basicity
        self.helicity = torch.FloatTensor(features[:, 20:40])     # Next 20 are helicity
        self.hydrophobicity = torch.FloatTensor(features[:, 40:60])   # Next 20 are hydrophobicity
        self.pi = torch.FloatTensor(features[:, 60:76])           # Next 16 are pI
        self.global_features = torch.FloatTensor(features[:, 76:]) # Last 2 are global features
        
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'basicity': self.basicity[idx],
            'helicity': self.helicity[idx],
            'hydrophobicity': self.hydrophobicity[idx],
            'pi': self.pi[idx],
            'global': self.global_features[idx]
        }, self.targets[idx]

# -----------------------
#     Model Definition
# -----------------------
class PeptideTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=2, num_encoder_layers=2, dropout=0.1):
        """
        Increased d_model to 256 for a bit more computational load.
        """
        super().__init__()
        
        # Property embeddings
        self.basicity_embedding = nn.Linear(20, d_model)
        self.helicity_embedding = nn.Linear(20, d_model)
        self.hydrophobicity_embedding = nn.Linear(20, d_model)
        self.pi_embedding = nn.Linear(16, d_model)
        self.global_embedding = nn.Linear(2, d_model)
        
        # Positional encoding (5 different feature types)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5, d_model))
        
        # Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 5, d_model),  # Combine all feature types
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, features_dict):
        # Create embeddings for each property type
        basicity_emb = self.basicity_embedding(features_dict['basicity'])
        helicity_emb = self.helicity_embedding(features_dict['helicity'])
        hydro_emb = self.hydrophobicity_embedding(features_dict['hydrophobicity'])
        pi_emb = self.pi_embedding(features_dict['pi'])
        global_emb = self.global_embedding(features_dict['global'])
        
        # Stack all embeddings => (batch_size, 5, d_model)
        x = torch.stack([
            basicity_emb, 
            helicity_emb, 
            hydro_emb, 
            pi_emb, 
            global_emb
        ], dim=1)
        
        # Add positional encodings
        x = x + self.pos_encoder
        
        # Apply transformer encoder => (batch_size, 5, d_model)
        transformed = self.transformer_encoder(x)
        
        # Flatten => (batch_size, 5 * d_model)
        flattened = transformed.reshape(transformed.size(0), -1)
        
        # Get final prediction => (batch_size, 1)
        output = self.output_projection(flattened)
        
        return output.squeeze(-1)

# -----------------------
#    Data Preparation
# -----------------------
def prepare_data(train_encoded, test_encoded, target='y_target', batch_size=2048):
    """Prepare data for training with scaling and DataLoader setup."""
    feature_cols = [col for col in train_encoded.columns 
                    if col not in ['spectrum_id', 'b_target', 'y_target']]
    
    X_train = train_encoded[feature_cols].values
    y_train = train_encoded[target].values
    X_test = test_encoded[feature_cols].values
    y_test = test_encoded[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = PeptideDataset(X_train_scaled, y_train)
    test_dataset = PeptideDataset(X_test_scaled, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,           # Increase number of workers
        pin_memory=True          # Pin memory for faster GPU transfers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,          # Also use multiple workers for test data
        pin_memory=True
    )
    
    return train_loader, test_loader, scaler

# -----------------------
#      Training Loop
# -----------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=5, device='cuda', early_stopping_patience=5):
    """Train model with early stopping + mixed-precision support."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    # Set up GradScaler for mixed-precision
    scaler = GradScaler(enabled=(device=='cuda')) 
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_actuals = []
        
        for batch_features, batch_targets in progress.track(train_loader, description=f"Epoch {epoch+1}"):
            # Move data to device
            batch_features = {k: v.to(device, non_blocking=True) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed-precision forward
            with autocast(device_type='cuda', enabled=(device=='cuda')):
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Record results
            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_actuals.extend(batch_targets.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = {k: v.to(device, non_blocking=True) for k, v in batch_features.items()}
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                # Mixed-precision forward in validation
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    outputs = model(batch_features)
                    v_loss = criterion(outputs, batch_targets)
                
                val_loss += v_loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        train_correlation = np.corrcoef(train_actuals, train_predictions)[0, 1]
        val_correlation = np.corrcoef(val_actuals, val_predictions)[0, 1]
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_correlation': train_correlation,
            'val_correlation': val_correlation
        })
        
        # Print progress
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {train_loss:.4f}, Correlation: {train_correlation:.4f}')
        print(f'  Validation Loss: {val_loss:.4f}, Correlation: {val_correlation:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    return pd.DataFrame(training_history)

# -----------------------
#      Evaluation
# -----------------------
def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and return predictions."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = {k: v.to(device, non_blocking=True) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                outputs = model(batch_features)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_targets.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)

# -----------------------
#      Plot Results
# -----------------------
def plot_results(history, predictions, actuals):
    """Plot training history and final predictions vs. actuals."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot losses
    ax1.plot(history['epoch'], history['train_loss'], label='Training Loss')
    ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot correlations
    ax2.plot(history['epoch'], history['train_correlation'], label='Training Correlation')
    ax2.plot(history['epoch'], history['val_correlation'], label='Validation Correlation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Training and Validation Correlation')
    ax2.legend()
    
    # Plot predictions vs actuals
    ax3.scatter(actuals, predictions, alpha=0.1)
    ax3.plot([0, 1], [0, 1], 'r--')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title(f'Predictions vs Actuals\nr={np.corrcoef(actuals, predictions)[0,1]:.3f}')
    
    plt.tight_layout()
    plt.show()

# -----------------------
#          Main
# -----------------------
def main():
    # Load prepared data
    print("Loading data...")
    train_encoded = pd.read_feather(
        "ftp://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomicsml/fragmentation/nist-humanhcd20160503-parsed-trainval-encoded.feather"
    )
    test_encoded = pd.read_feather(
        "ftp://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomicsml/fragmentation/nist-humanhcd20160503-parsed-test-encoded.feather"
    )

    # Prepare data
    print("Preparing data...")
    train_loader, test_loader, scaler = prepare_data(
        train_encoded, test_encoded,
        target='y_target',
        batch_size=2048  # Large batch size to utilize the GPU more
    )
    
    # Initialize model + optimizer + loss
    print("Initializing transformer model...")
    model = PeptideTransformer(
        d_model=256,   # Increased dimension
        nhead=2,
        num_encoder_layers=2,
        dropout=0.1
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    print("Training model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        device=device,
        early_stopping_patience=5
    )
    
    # Load best model and evaluate
    print("Evaluating model...")
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    predictions, actuals = evaluate_model(model, test_loader, device)
    
    # Plot results
    print("Plotting results...")
    plot_results(history, predictions, actuals)
    
    # Print final correlation
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    print(f'\nFinal Test Set Correlation: {correlation:.4f}')
    
    return predictions, actuals, history

if __name__ == "__main__":
    predictions, actuals, history = main()
