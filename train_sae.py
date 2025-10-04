import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return confi
config = load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = config["general"]["cuda_devices"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =================== Custom Modules ===================

# Custom JumpReLU activation
class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()

# Apply top-k mask to enforce sparsity
def topk_mask(tensor, k, topk_range=None):
    device = tensor.device
    batch_size, dim = tensor.shape
    mask = torch.zeros_like(tensor, device=device)

    if topk_range is not None:
        start_idx, end_idx = topk_range
        sub_tensor = tensor[:, start_idx:end_idx]
        topk_vals, topk_indices = torch.topk(sub_tensor.abs(), k, dim=1)
        topk_indices += start_idx  # shift indices to match full tensor
        mask.scatter_(1, topk_indices, 1)
    else:
        # Apply top-k to entire tensor if no range specified
        topk_vals, topk_indices = torch.topk(tensor.abs(), k, dim=1)
        mask.scatter_(1, topk_indices, 1)

    return tensor * mask

# =================== Dataset Loader ===================

class ImageEmbeddingsDataset(Dataset):
    def __init__(self, npz_path=None):
        # Use provided path or default to the specific file
        if npz_path is None:
            npz_path = ""
        
        # Load the NPZ file
        try:
            data = np.load(npz_path)
            # Assuming the embeddings are stored with a specific key
            # Common keys are 'embeddings', 'features', 'X', or 'arr_0'
            # You may need to adjust based on your NPZ file structure
            print(data.keys())
            embeddings = data['combined_embeddings']
            print(f"Loaded embeddings shape: {embeddings.shape}")
            
            # Convert to torch tensor
            self.X = torch.from_numpy(embeddings).float()
            print(f"Loaded embeddings from {npz_path}")
            print(f"Embeddings shape: {self.X.shape}")
            
        except Exception as e:
            print(f"Failed to load embeddings from {npz_path}: {e}")
            raise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# =================== Model Definition ===================

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2048, topk=None, jump_params=None):
        super(SparseAutoencoder, self).__init__()
        self.topk = topk
        gamma, beta = jump_params if jump_params else (1.0, 1.0)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            JumpReLU(gamma=gamma, beta=beta)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        if self.topk:
            # Apply top-k sparsity (you can adjust topk_range as needed)
            z = topk_mask(z, self.topk, topk_range=(32, -1))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x):
        z = self.encoder(x)
        if self.topk:
            z = topk_mask(z, self.topk, topk_range=(32, -1))
        return z

# =================== Training & Encoding ===================

def train_sparse_autoencoder(model, dataloader, num_epochs, learning_rate, model_save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) as pbar:
            for x in pbar:
                x = x.to(device)
                model.to(device)

                x_reconstructed, z = model(x)
                
                # Reconstruction loss
                loss = criterion(x_reconstructed, x)
                
                # Optional: Add L1 regularization for sparsity
                if config["training"].get("l1_weight", 0) > 0:
                    l1_loss = config["training"]["l1_weight"] * z.abs().mean()
                    loss = loss + l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
        
        if epoch % config["logging"]["printing_every"] == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

def encode_and_save(model, dataloader, save_path):
    model.eval()
    all_z = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            z = model.encode(x)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    torch.save(all_z, save_path)
    print(f"Saved encoded embeddings to {save_path}")

# =================== Runner ===================

def train_image_sae(model_num, npz_path=None):
    embeddings_dataset = ImageEmbeddingsDataset(npz_path)
    train_loader = DataLoader(
        dataset=embeddings_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True
    )

    input_dim = embeddings_dataset[0].shape[0]
    print(f"Input dimension: {input_dim}")

    num_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    model_save_path = config["general"]["model_save_path"] + model_num + '.pth'
    latent_dim = config["training"]["sparse_dim"]
    topk = config["training"]["topk"]  # Top-k sparsity
    jump_params = config["training"].get("jump_params", (1.0, 1.0))  # gamma, beta

    model = SparseAutoencoder(input_dim, latent_dim, topk, jump_params).to(device)
    model = train_sparse_autoencoder(model, train_loader, num_epochs, learning_rate, model_save_path)
    
    # Optional: Save encoded representations after training
    if config["general"].get("save_encodings", False):
        encode_save_path = config["general"]["encodings_save_path"] + model_num + '.pt'
        encode_and_save(model, train_loader, encode_save_path)

if __name__ == "__main__":
    # model_num = config['general']['model_num']
    # Optional: specify custom NPZ path in config
    npz_path = config["general"].get("npz_path", None)
    for model_num in range(2,100):
        print(f"Training SAE model: {model_num}")
        train_image_sae(str(model_num), npz_path)