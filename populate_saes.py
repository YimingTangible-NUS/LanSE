import yaml
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shutil
from tqdm import tqdm
from PIL import ImageFile
import random
from pathlib import Path
from torchvision import transforms, models
from open_clip import create_model_from_pretrained, get_tokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_TEXT_CHUNK = 100 * (1024 ** 2)

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = config["general"]["cuda_devices"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =================== Custom Modules (from training script) ===================

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()

def topk_mask(tensor, k, topk_range=None):
    device = tensor.device
    batch_size, dim = tensor.shape
    mask = torch.zeros_like(tensor, device=device)

    if topk_range is not None:
        start_idx, end_idx = topk_range
        sub_tensor = tensor[:, start_idx:end_idx]
        topk_vals, topk_indices = torch.topk(sub_tensor.abs(), k, dim=1)
        topk_indices += start_idx
        mask.scatter_(1, topk_indices, 1)
    else:
        topk_vals, topk_indices = torch.topk(tensor.abs(), k, dim=1)
        mask.scatter_(1, topk_indices, 1)

    return tensor * mask

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
            z = topk_mask(z, self.topk, topk_range=(32, -1))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x):
        z = self.encoder(x)
        if self.topk:
            z = topk_mask(z, self.topk, topk_range=(32, -1))
        return z

# =================== Dataset for Paired Image-Text Loading ===================

class ChestXRayPairedDataset(Dataset):
    def __init__(self, data_folders, max_pairs_per_folder=None):
        self.data_pairs = []
        
        for folder_path in data_folders:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"Warning: {folder_path} does not exist")
                continue
            
            # Collect all files
            all_files = list(folder.glob('*'))
            
            # Create a dictionary to match basenames
            file_dict = {}
            for file_path in all_files:
                if file_path.is_file():
                    basename = file_path.stem  # Get filename without extension
                    ext = file_path.suffix.lower()
                    
                    if basename not in file_dict:
                        file_dict[basename] = {}
                    
                    if ext == '.txt':
                        file_dict[basename]['text'] = str(file_path)
                    elif ext in ['.png', '.jpg', '.jpeg']:
                        file_dict[basename]['image'] = str(file_path)
            
            # Create pairs
            pairs = []
            for basename, files in file_dict.items():
                if 'image' in files and 'text' in files:
                    pairs.append((files['image'], files['text']))
                # elif 'image' in files:
                #     # If no text file, we'll handle it later
                #     pairs.append((files['image'], None))
            
            # Limit number of pairs if specified
            if max_pairs_per_folder and len(pairs) > max_pairs_per_folder:
                pairs = random.sample(pairs, max_pairs_per_folder)
            
            self.data_pairs.extend(pairs)
            print(f"Found {len(pairs)} image-text pairs in {folder_path}")
        
        print(f"Total pairs: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, text_path = self.data_pairs[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Load text
        text = ""
        if text_path and os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                print(f"Error loading {text_path}: {e}")
        
        return image, text, image_path

# =================== Feature Extraction with BiomedCLIP ===================

class BiomedCLIPFeatureExtractor:
    def __init__(self):
        print("Loading BiomedCLIP model...")
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model.to(device)
        self.model.eval()
        
        # Get embedding dimensions
        self.image_dim = 512  # BiomedCLIP image embedding dimension
        self.text_dim = 512   # BiomedCLIP text embedding dimension
        self.combined_dim = self.image_dim + self.text_dim
    
    def extract_features(self, images, texts):
        """Extract and concatenate image and text features"""
        with torch.no_grad():
            # Process images
            image_tensors = torch.stack([self.preprocess(img) for img in images]).to(device)
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Process texts
            text_tokens = self.tokenizer(texts, context_length=256).to(device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Concatenate features
            combined_features = torch.cat([image_features, text_features], dim=1)
            
        return combined_features

def load_combined_embeddings_from_npz(npz_path):
    """Load precomputed combined embeddings from NPZ file"""
    data = np.load(npz_path)
    
    # Try different possible keys for combined embeddings
    if 'combined_embeddings' in data:
        embeddings = data['combined_embeddings']
    elif 'embeddings' in data:
        embeddings = data['embeddings']
    else:
        # If separate embeddings exist, concatenate them
        if 'image_embeddings' in data and 'text_embeddings' in data:
            image_emb = data['image_embeddings']
            text_emb = data['text_embeddings']
            embeddings = np.concatenate([image_emb, text_emb], axis=1)
        else:
            raise ValueError("Could not find combined embeddings in NPZ file")
    
    # Load paths if available
    paths = data.get('paths', None)
    
    return torch.from_numpy(embeddings).float(), paths

# =================== Main Population Function ===================

def populate_features(model_num, use_precomputed_embeddings=False):
    torch.backends.cudnn.benchmark = True

    # Data folders with new structure
    data_folders = [
        ""
    ]

    if use_precomputed_embeddings:
        # Load precomputed combined embeddings
        npz_path = config["general"].get("combined_embeddings_path", 
                                         "cxr_combined_embeddings.npz")
        embeddings, image_paths = load_combined_embeddings_from_npz(npz_path)
        input_dim = embeddings.shape[1]
        print(f"Loaded {len(embeddings)} precomputed combined embeddings with dimension {input_dim}")
        
    else:
        # Initialize BiomedCLIP for on-the-fly feature extraction
        feature_extractor = BiomedCLIPFeatureExtractor()
        input_dim = feature_extractor.combined_dim
        print(f"Using BiomedCLIP with combined embedding dimension: {input_dim}")
    
    # Load SAE model
    model_save_path = config["general"]["model_save_path"] + model_num + '.pth'
    latent_dim = config["training"]["sparse_dim"]
    topk = config["training"].get("topk", None)
    jump_params = config["training"].get("jump_params", (1.0, 1.0))
    
    sparse_model = SparseAutoencoder(input_dim, latent_dim, topk, jump_params).to(device)
    sparse_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    sparse_model.eval()

    # Configuration
    THRESHOLD = config['general']['threshold']
    OUTPUT_DIR = f"./features/features_model{model_num}_combined"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MAX_IMAGES_PER_FEATURE = config['general'].get('max_images_per_feature', 20)
    BATCH_SIZE = config['general'].get('populate_batch_size', 32)

    # Track how many images per feature
    feature_image_counts = {}

    if use_precomputed_embeddings:
        # Process precomputed embeddings
        print("Processing precomputed combined embeddings...")
        
        # Collect all available image-text pairs from the data folders
        all_pairs = []
        for folder_path in data_folders:
            folder = Path(folder_path)
            if folder.exists():
                # Collect files by basename
                file_dict = {}
                for file_path in folder.glob('*'):
                    if file_path.is_file():
                        basename = file_path.stem
                        ext = file_path.suffix.lower()
                        
                        if basename not in file_dict:
                            file_dict[basename] = {}
                        
                        if ext == '.txt':
                            file_dict[basename]['text'] = str(file_path)
                        elif ext in ['.png', '.jpg', '.jpeg']:
                            file_dict[basename]['image'] = str(file_path)
                
                # Create pairs
                for basename, files in file_dict.items():
                    if 'image' in files:
                        all_pairs.append((files.get('image'), files.get('text')))
        
        print(f"Found {len(all_pairs)} total image-text pairs available")
        
        # Match with paths from NPZ if available
        if image_paths is None or len(image_paths) == 0:
            image_paths = [p[0] for p in all_pairs[:len(embeddings)]]
            text_paths = [p[1] for p in all_pairs[:len(embeddings)]]
        else:
            # Parse paths to get text paths (assuming they follow naming convention)
            text_paths = []
            for img_path in image_paths:
                if isinstance(img_path, (str, bytes)):
                    img_path = str(img_path)
                    text_path = img_path.rsplit('.', 1)[0] + '.txt'
                    text_paths.append(text_path if os.path.exists(text_path) else None)
                else:
                    text_paths.append(None)
        
        # Process in batches
        num_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc="Processing embeddings"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(embeddings))
            
            batch_embeddings = embeddings[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                # Encode through SAE
                z_sparse = sparse_model.encode(batch_embeddings)
                
                # Find activated features (above threshold)
                z_mask = (z_sparse > THRESHOLD).cpu()
            
            # Save images and texts for activated features
            for idx_in_batch, activated_feature_idxs in enumerate(z_mask):
                global_idx = start_idx + idx_in_batch
                
                for feature_idx in activated_feature_idxs.nonzero(as_tuple=True)[0]:
                    feature_idx = feature_idx.item()
                    
                    if feature_image_counts.get(feature_idx, 0) >= MAX_IMAGES_PER_FEATURE:
                        continue
                    
                    # Create feature directory
                    feature_dir = os.path.join(OUTPUT_DIR, f"feature_{feature_idx}")
                    os.makedirs(feature_dir, exist_ok=True)
                    
                    # Save the image and text if available
                    if global_idx < len(image_paths):
                        img_path = image_paths[global_idx]
                        txt_path = text_paths[global_idx] if global_idx < len(text_paths) else None
                        
                        if isinstance(img_path, (str, bytes)):
                            img_path = str(img_path)
                            
                            # Find the actual file
                            if not os.path.isabs(img_path):
                                for folder in data_folders:
                                    potential_path = os.path.join(folder, img_path)
                                    if os.path.exists(potential_path):
                                        img_path = potential_path
                                        break
                            
                            if os.path.exists(img_path):
                                basename = os.path.splitext(os.path.basename(img_path))[0]
                                
                                try:
                                    # Save image
                                    img = Image.open(img_path).convert("RGB")
                                    img_save_path = os.path.join(feature_dir, f"{basename}.png")
                                    img.save(img_save_path)
                                    
                                    # Save text if available
                                    if txt_path and os.path.exists(txt_path):
                                        txt_save_path = os.path.join(feature_dir, f"{basename}.txt")
                                        shutil.copy(txt_path, txt_save_path)
                                    
                                    # Update count
                                    feature_image_counts[feature_idx] = feature_image_counts.get(feature_idx, 0) + 1
                                    
                                    # Save activation value
                                    activation_value = z_sparse[idx_in_batch, feature_idx].item()
                                    with open(os.path.join(feature_dir, "activations.txt"), "a") as f:
                                        f.write(f"{basename}: {activation_value:.4f}\n")
                                    
                                except Exception as e:
                                    print(f"Error processing files for {basename}: {e}")
    
    else:
        # Process images and texts directly using BiomedCLIP
        dataset = ChestXRayPairedDataset(
            data_folders, 
            max_pairs_per_folder=config['general'].get('max_images_per_folder', None)
        )
        
        def collate_fn(batch):
            images = []
            texts = []
            paths = []
            for img, text, img_path in batch:
                images.append(img)
                texts.append(text if text else "")  # Use empty string if no text
                paths.append(img_path)
            return images, texts, paths
        
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Process batches
        for batch_data in tqdm(data_loader, desc="Processing image-text pairs"):
            images, texts, image_paths = batch_data
            
            if not images:
                continue

            with torch.no_grad():
                # Get combined features from BiomedCLIP
                combined_features = feature_extractor.extract_features(images, texts)
                
                # Encode through SAE
                z_sparse = sparse_model.encode(combined_features)
                
                # Find activated features (above threshold)
                z_mask = (z_sparse > THRESHOLD).cpu()

            # Save images and texts for activated features
            for idx, activated_feature_idxs in enumerate(z_mask):
                image_path = image_paths[idx]
                text = texts[idx]
                
                for feature_idx in activated_feature_idxs.nonzero(as_tuple=True)[0]:
                    feature_idx = feature_idx.item()
                    
                    if feature_image_counts.get(feature_idx, 0) >= MAX_IMAGES_PER_FEATURE:
                        continue
                    
                    # Create feature directory
                    feature_dir = os.path.join(OUTPUT_DIR, f"feature_{feature_idx}")
                    os.makedirs(feature_dir, exist_ok=True)
                    
                    # Generate filenames
                    basename = os.path.splitext(os.path.basename(image_path))[0]
                    img_save_path = os.path.join(feature_dir, f"{basename}.png")
                    txt_save_path = os.path.join(feature_dir, f"{basename}.txt")
                    
                    try:
                        # Save the image
                        shutil.copy(image_path, img_save_path)
                        
                        # Save the text
                        if text:
                            with open(txt_save_path, 'w', encoding='utf-8') as f:
                                f.write(text)
                        
                        # Update count
                        feature_image_counts[feature_idx] = feature_image_counts.get(feature_idx, 0) + 1
                        
                        # Save activation value
                        activation_value = z_sparse[idx, feature_idx].item()
                        with open(os.path.join(feature_dir, "activations.txt"), "a") as f:
                            f.write(f"{basename}: {activation_value:.4f}\n")
                            
                    except Exception as e:
                        print(f"Error saving {basename}: {e}")

    # Summary statistics
    print(f"\nPopulation complete!")
    print(f"Total features with saved samples: {len(feature_image_counts)}")
    if feature_image_counts:
        print(f"Average samples per feature: {np.mean(list(feature_image_counts.values())):.2f}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_num}\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"Embedding type: Combined (Image + Text)\n")
        f.write(f"Embedding dimension: {input_dim}\n")
        f.write(f"Total features with activations: {len(feature_image_counts)}\n")
        f.write(f"Feature activation counts:\n")
        for feat_idx, count in sorted(feature_image_counts.items()):
            f.write(f"  Feature {feat_idx}: {count} samples\n")

if __name__ == '__main__':  
    # model_num = config['general']['model_num']
    # Set to True to use precomputed embeddings from NPZ file
    # Set to False to extract features from images/texts directly
    use_precomputed = config['general'].get('use_precomputed_embeddings', True)
    for model_num in range(2,100):
        print(f"\nPopulating features using SAE model: {model_num}")
        populate_features(str(model_num), use_precomputed_embeddings=use_precomputed)