import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Load BiomedCLIP model
print("Loading BiomedCLIP model...")
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.to(device)
model.eval()

# === Dataset Class for CXR Images with Text ===
class CXRImageTextDataset(Dataset):
    def __init__(self, data_pairs, preprocess, tokenizer):
        """
        Args:
            data_pairs: List of tuples (image_path, text_path, label)
            preprocess: Preprocessing function from BiomedCLIP
            tokenizer: Tokenizer from BiomedCLIP
        """
        self.data_pairs = data_pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_path, text_path, label = self.data_pairs[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        
        # Load and tokenize text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Tokenize text
        text_tokens = self.tokenizer([text], context_length=256)
        
        return image_tensor, text_tokens, label, image_path

def collect_data_pairs(folder_path, label):
    """Collect paired image and text files from folder"""
    data_pairs = []
    
    # Get all files in the folder
    all_files = os.listdir(folder_path)
    
    # Create a dictionary to match basenames
    file_dict = {}
    for file in all_files:
        basename, ext = os.path.splitext(file)
        if ext.lower() in ['.txt']:
            if basename not in file_dict:
                file_dict[basename] = {}
            file_dict[basename]['text'] = os.path.join(folder_path, file)
            file_dict[basename]['image'] = os.path.join(folder_path, file.replace('.txt', '.png'))  # Assuming images are PNGs
    
    # Create pairs
    for basename, files in file_dict.items():
        if 'image' in files and 'text' in files:
            data_pairs.append((files['image'], files['text'], label))
        elif 'image' in files:
            # If no text file, create a dummy one
            print(f"Warning: No text file for {basename}, using empty text")
            data_pairs.append((files['image'], None, label))
    
    return data_pairs

def extract_embeddings(model, dataloader, device):
    """Extract both image and text embeddings using BiomedCLIP"""
    all_image_embeddings = []
    all_text_embeddings = []
    all_combined_embeddings = []
    all_labels = []
    all_paths = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images, text_tokens, labels, paths = batch
            images = images.to(device)
            
            # Handle text tokens
            text_list = []
            for i in range(len(text_tokens[0])):
                text_list.append(text_tokens[0][i])
            text_tokens_tensor = torch.stack(text_list).to(device)
            
            # Get image features (embeddings)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
            
            # Get text features (embeddings)
            text_features = model.encode_text(text_tokens_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
            
            # Concatenate image and text embeddings
            combined_features = torch.cat([image_features, text_features], dim=1)
            
            all_image_embeddings.append(image_features.cpu().numpy())
            all_text_embeddings.append(text_features.cpu().numpy())
            all_combined_embeddings.append(combined_features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    # Concatenate all embeddings
    all_image_embeddings = np.vstack(all_image_embeddings)
    all_text_embeddings = np.vstack(all_text_embeddings)
    all_combined_embeddings = np.vstack(all_combined_embeddings)
    all_labels = np.array(all_labels)
    
    return all_image_embeddings, all_text_embeddings, all_combined_embeddings, all_labels, all_paths

def collate_fn(batch):
    """Custom collate function to handle variable-length text"""
    images = torch.stack([item[0] for item in batch])
    
    # Collect all text tokens
    texts = [item[1] for item in batch]
    max_len = max(t.shape[1] if t is not None else 0 for t in texts)
    
    # Pad text tokens to same length
    padded_texts = []
    for t in texts:
        if t is not None:
            if t.shape[1] < max_len:
                padding = torch.zeros((1, max_len - t.shape[1]), dtype=t.dtype)
                t = torch.cat([t, padding], dim=1)
            padded_texts.append(t.squeeze(0))
        else:
            # Create empty text token if no text file
            padded_texts.append(torch.zeros(max_len, dtype=torch.long))
    
    text_tokens = [torch.stack(padded_texts)]
    
    labels = torch.tensor([item[2] for item in batch])
    paths = [item[3] for item in batch]
    
    return images, text_tokens, labels, paths

def main():
    # === Collect Real MIMIC-CXR Images and Text ===
    print("Collecting real MIMIC-CXR data...")
    real_folder = ""
    real_pairs = collect_data_pairs(real_folder, label=1)  # Label 1 for real
    print(f"Found {len(real_pairs)} real image-text pairs")
    
    # === Collect Generated Images and Text ===
    print("Collecting generated data...")
    generated_folder = ""
    generated_pairs = collect_data_pairs(generated_folder, label=0)  # Label 0 for generated
    print(f"Found {len(generated_pairs)} generated image-text pairs")
    
    # === Combine all data ===
    all_data_pairs = real_pairs + generated_pairs
    print(f"Total pairs to process: {len(all_data_pairs)}")
    
    # === Create Dataset and DataLoader ===
    dataset = CXRImageTextDataset(all_data_pairs, preprocess, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=32,  # Adjust based on GPU memory
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # === Extract Embeddings ===
    print("Extracting embeddings...")
    image_embeddings, text_embeddings, combined_embeddings, labels, paths = extract_embeddings(
        model, dataloader, device
    )
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # === Save Embeddings ===
    # Save all embeddings including the concatenated ones
    np.savez_compressed(
        "cxr_embeddings_biomedclip_combined.npz",
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        combined_embeddings=combined_embeddings,  # Concatenated image + text
        labels=labels,
        paths=np.array(paths)
    )
    print("Saved embeddings to cxr_embeddings_biomedclip_combined.npz")
    
    # Also save just the combined embeddings in a separate file for convenience
    np.savez_compressed(
        "cxr_combined_embeddings.npz",
        embeddings=combined_embeddings,  # This is the concatenated version
        labels=labels,
        paths=np.array(paths)
    )
    print("Saved combined embeddings to cxr_combined_embeddings.npz")
    
    print("\nEmbedding extraction complete!")
    
    # === Print Statistics ===
    print("\n=== Statistics ===")
    print(f"Real data pairs: {np.sum(labels == 1)}")
    print(f"Generated data pairs: {np.sum(labels == 0)}")
    print(f"Image embedding dimension: {image_embeddings.shape[1]}")
    print(f"Text embedding dimension: {text_embeddings.shape[1]}")
    print(f"Combined embedding dimension: {combined_embeddings.shape[1]}")
    
    return combined_embeddings, labels, paths

if __name__ == "__main__":
    embeddings, labels, paths = main()
    
    # Optional: Quick verification
    print("\n=== Verification ===")
    print("Testing similarity between first 5 combined embeddings...")
    similarity_matrix = np.dot(embeddings[:5], embeddings[:5].T)
    print("Similarity matrix:")
    print(similarity_matrix)