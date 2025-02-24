import os
import pickle
import numpy as np
import pandas as pd
from rich import print, progress

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


# Device Selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")


# Data Preparaion
def read_msp(filename):
    spectrum = {}
    mz = []
    intensity = []
    annotation = []

    with progress.open(filename, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Name: "):
                if spectrum:
                    # Finalize previous spectrum
                    spectrum["sequence"] = spectrum["Fullname"].split(".")[1]
                    spectrum["mz"] = np.array(mz, dtype="float32")
                    spectrum["intensity"] = np.array(intensity, dtype="float32")
                    spectrum["annotation"] = np.array(annotation, dtype="str")
                    yield spectrum

                    # Reset for the next spectrum
                    spectrum = {}
                    mz = []
                    intensity = []
                    annotation = []
                spectrum["Name"] = line[6:].strip()

            elif line.startswith("Comment: "):
                metadata = [item.split("=") for item in line[9:].split(" ")]
                for item in metadata:
                    if len(item) == 2:
                        spectrum[item[0]] = item[1]
            elif line.startswith("Num peaks: "):
                spectrum["Num peaks"] = int(line[11:].strip())
            elif "\t" in line and len(line.split("\t")) == 3:
                parts = line.split("\t")
                mz.append(parts[0])
                intensity.append(parts[1])
                annotation.append(parts[2].strip('"'))

    # Final leftover spectrum
    if spectrum:
        spectrum["sequence"] = spectrum["Fullname"].split(".")[1]
        spectrum["mz"] = np.array(mz, dtype="float32")
        spectrum["intensity"] = np.array(intensity, dtype="float32")
        spectrum["annotation"] = np.array(annotation, dtype="str")
        yield spectrum


# Dataloader 
class DeNovoPeptideDataset(Dataset):
    def __init__(self, msp_filename, max_mz=2000, max_length=35):
        self.samples = list(read_msp(msp_filename))
        self.max_mz = max_mz
        self.max_length = max_length
        
        # 20 amino acids + blank token
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.token_to_index = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.token_to_index["-"] = 20
        self.index_to_token = {i: aa for aa, i in self.token_to_index.items()}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        precursor_mz = float(sample.get("Parent", 0))
        charge = int(sample.get("Charge", 0))
        
        input_vector = np.array([precursor_mz, charge], dtype=np.float32)
        
        # Target sequence
        sequence = sample["sequence"]
        # Assume no modifications => offset=0 for each residue
        target_offsets = [0.0] * len(sequence)
        
        # Convert sequence to token indices
        target_tokens = [self.token_to_index.get(aa, 20) for aa in sequence]
        
        # Pad/truncate to max_length
        if len(target_tokens) < self.max_length:
            pad_len = self.max_length - len(target_tokens)
            target_tokens += [20] * pad_len
            target_offsets += [0.0] * pad_len
        else:
            target_tokens = target_tokens[:self.max_length]
            target_offsets = target_offsets[:self.max_length]
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_vector)
        target_tokens_tensor = torch.LongTensor(target_tokens)
        target_offsets_tensor = torch.FloatTensor(target_offsets)
        
        return input_tensor, (target_tokens_tensor, target_offsets_tensor)


# Transformer Model
class DeNovoPeptideSequencer(nn.Module):
    def __init__(self, input_dim, num_tokens=21, max_length=35, 
                 d_model=256, nhead=4, num_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.max_length = max_length
        self.d_model = d_model
        
        # Simple MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Learnable decoder input embeddings for each position
        self.decoder_input = nn.Parameter(torch.randn(1, max_length, d_model))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output heads
        self.residue_head = nn.Linear(d_model, num_tokens)  # classification
        self.ptm_head = nn.Linear(d_model, 1)              # offset regression
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        enc_out = self.encoder(x)  # (batch_size, d_model)
        memory = enc_out.unsqueeze(1).repeat(1, self.max_length, 1)  # (batch_size, max_length, d_model)
        
        dec_in = self.decoder_input.expand(x.size(0), -1, -1)  # (batch_size, max_length, d_model)
        decoded = self.transformer_decoder(tgt=dec_in, memory=memory)  # (batch_size, max_length, d_model)
        
        residue_logits = self.residue_head(decoded)    # (batch_size, max_length, num_tokens)
        ptm_offsets = self.ptm_head(decoded).squeeze(-1)  # (batch_size, max_length)
        return residue_logits, ptm_offsets


# Training Loop
def train_model(model, dataloader, optimizer, device, num_epochs=10, alpha=1.0):
    
    model.to(device)
    model.train()
    
    # Enable mixed precision only if device.type == 'cuda'
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        
        for inputs, (target_tokens, target_offsets) in progress.track(dataloader, description=f"Epoch {epoch+1}"):
            inputs = inputs.to(device, non_blocking=True)
            target_tokens = target_tokens.to(device, non_blocking=True)
            target_offsets = target_offsets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            # Autocast only if on CUDA
            with autocast(enabled=(device.type == 'cuda')):
                residue_logits, ptm_offsets = model(inputs)
                ce_loss = ce_loss_fn(
                    residue_logits.view(-1, residue_logits.size(-1)), 
                    target_tokens.view(-1)
                )
                mse_loss = mse_loss_fn(ptm_offsets, target_offsets)
                loss = ce_loss + alpha * mse_loss
            
            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * inputs.size(0)
            
            # Compute token accuracy
            preds = residue_logits.argmax(dim=-1)
            correct_tokens += (preds == target_tokens).sum().item()
            total_tokens += target_tokens.numel()
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        accuracy = correct_tokens / total_tokens
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Token Accuracy = {accuracy:.4f}")
        history.append({"epoch": epoch+1, "loss": avg_loss, "accuracy": accuracy})
    
    return history


# Evaluation 
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds_tokens = []
    all_preds_offsets = []
    all_targets_tokens = []
    all_targets_offsets = []
    
    with torch.no_grad():
        for inputs, (target_tokens, target_offsets) in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            target_tokens = target_tokens.to(device, non_blocking=True)
            target_offsets = target_offsets.to(device, non_blocking=True)
            
            # No autocast for evaluation, or only if device.type == 'cuda'
            with autocast(enabled=(device.type == 'cuda')):
                residue_logits, ptm_offsets = model(inputs)
            preds_tokens = residue_logits.argmax(dim=-1)
            
            all_preds_tokens.append(preds_tokens.cpu())
            all_preds_offsets.append(ptm_offsets.cpu())
            all_targets_tokens.append(target_tokens.cpu())
            all_targets_offsets.append(target_offsets.cpu())
    
    all_preds_tokens = torch.cat(all_preds_tokens, dim=0)
    all_preds_offsets = torch.cat(all_preds_offsets, dim=0)
    all_targets_tokens = torch.cat(all_targets_tokens, dim=0)
    all_targets_offsets = torch.cat(all_targets_offsets, dim=0)
    
    return (all_preds_tokens, all_preds_offsets), (all_targets_tokens, all_targets_offsets)


def main():
    msp_file = "datasets/human_hcd_tryp_best.msp"
    
    # Build dataset
    dataset = DeNovoPeptideDataset(msp_file, max_mz=2000, max_length=35)
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Split
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, train_size=0.9, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=0, pin_memory=False)
    
    # Model
    input_dim = 2
    model = DeNovoPeptideSequencer(
        input_dim=input_dim, 
        num_tokens=21, 
        max_length=35, 
        d_model=256, 
        nhead=4, 
        num_decoder_layers=3, 
        dropout=0.1
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    history = train_model(model, train_loader, optimizer, device=device, num_epochs=10, alpha=1.0)
    
    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/best_de_novo_model.pth")
    print("Model saved to results/best_de_novo_model.pth")
    
    # Evaluate
    (pred_tokens, pred_offsets), (true_tokens, true_offsets) = evaluate_model(model, test_loader, device=device)
    
    # Decode some predictions
    dataset_dummy = DeNovoPeptideDataset(msp_file)
    index_to_token = dataset_dummy.index_to_token
    
    def decode_tokens(token_tensor):
        return "".join([index_to_token[token.item()] for token in token_tensor])
    
    print("\nSample predictions on test set:")
    for i in range(5):
        pred_seq = decode_tokens(pred_tokens[i])
        true_seq = decode_tokens(true_tokens[i])
        print(f"True: {true_seq}  |  Predicted: {pred_seq}")
    
    # Save evaluation
    results = {
        "history": history,
        "pred_tokens": pred_tokens,
        "pred_offsets": pred_offsets,
        "true_tokens": true_tokens,
        "true_offsets": true_offsets
    }
    with open("results/evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Evaluation results saved to results/evaluation_results.pkl")


if __name__ == "__main__":
    main()
