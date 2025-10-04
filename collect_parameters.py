import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from PIL import Image

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# =================== Model Components ===================

class JumpReLU(nn.Module):
    """Custom JumpReLU activation function"""
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()

def topk_mask(tensor, k, topk_range=None):
    """Apply top-k masking to tensor"""
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
        topk_vals, topk_indices = torch.topk(tensor.abs(), k, dim=1)
        mask.scatter_(1, topk_indices, 1)

    return tensor * mask

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder model structure"""
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

class FeatureProbe(nn.Module):
    """Feature probe wrapper for combining multiple feature detectors"""
    def __init__(self, probes):
        super().__init__()
        self.probes = nn.ModuleList(probes)

    def forward(self, z):
        return torch.cat([probe(z) for probe in self.probes], dim=1)

# =================== Loading Functions ===================

def load_accuracy_summary(model_num: str, base_path: str = "") -> Dict:
    """Load the accuracy summary for a model"""
    summary_path = os.path.join(base_path, f"accuracy_summary_model{model_num}.json")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Accuracy summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary

def load_sae_model(model_num: str, base_path: str = "") -> SparseAutoencoder:
    """Load a trained SAE model"""
    model_path = os.path.join(base_path, f"{model_num}.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Get model configuration from config file
    input_dim = config.get("training", {}).get("input_dim", 1024)  # Combined embedding dimension
    latent_dim = config.get("training", {}).get("sparse_dim", 2048)
    topk = config.get("training", {}).get("topk", 32)
    jump_params = config.get("training", {}).get("jump_params", (1.0, 1.0))
    
    # Create and load model
    model = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        topk=topk,
        jump_params=jump_params
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def extract_high_accuracy_features(summary: Dict, accuracy_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Extract features with accuracy above threshold
    
    Returns:
        List of tuples (feature_num, explanation, accuracy)
    """
    high_accuracy_features = []
    
    feature_accuracies = summary.get("feature_accuracies", {})
    
    for feature_num, feature_data in feature_accuracies.items():
        accuracy = feature_data.get("accuracy", 0)
        
        if accuracy >= accuracy_threshold:
            explanation = feature_data.get("explanation", "")
            high_accuracy_features.append((feature_num, explanation, accuracy))
    
    # Sort by accuracy (descending)
    high_accuracy_features.sort(key=lambda x: x[2], reverse=True)
    
    return high_accuracy_features

def create_feature_probes(model: SparseAutoencoder, feature_indices: List[int]) -> List[nn.Module]:
    """Create probe modules for specified feature indices"""
    probes = []
    
    # Get the encoder linear layer (first layer of encoder sequential)
    encoder_linear = model.encoder[0]
    
    for feature_idx in feature_indices:
        # Extract weights and bias for this feature
        w = encoder_linear.weight[feature_idx].unsqueeze(0).clone().detach()
        b = encoder_linear.bias[feature_idx].unsqueeze(0).clone().detach()
        
        # Create a probe (single linear layer)
        probe = nn.Linear(w.shape[1], 1)
        probe.weight.data = w
        probe.bias.data = b
        
        probes.append(probe)
    
    return probes

# =================== Main Function ===================

def create_lanse_model(
    model_nums: List[str],
    accuracy_threshold: float = 0.7,
    max_features_per_model: int = None,
    output_path: str = None,
    save_metadata: bool = True
):
    """Create a LanSE model from high-accuracy features across multiple models
    
    Args:
        model_nums: List of model numbers to process
        accuracy_threshold: Minimum accuracy for feature inclusion
        max_features_per_model: Maximum features to take from each model
        output_path: Where to save the LanSE model
        save_metadata: Whether to save metadata about included features
    """
    
    print(f"Creating LanSE model from high-accuracy features")
    print(f"Models: {model_nums}")
    print(f"Accuracy threshold: {accuracy_threshold}")
    print(f"Max features per model: {max_features_per_model or 'unlimited'}")
    print("-" * 50)
    
    all_probes = []
    metadata = {
        "models": {},
        "total_features": 0,
        "accuracy_threshold": accuracy_threshold,
        "feature_details": []
    }
    
    # Process each model
    for model_num in model_nums:
        print(f"\nProcessing Model {model_num}...")
        
        try:
            # Load accuracy summary
            summary = load_accuracy_summary(model_num)
            
            # Extract high-accuracy features
            high_acc_features = extract_high_accuracy_features(summary, accuracy_threshold)
            
            if not high_acc_features:
                print(f"  No features above {accuracy_threshold} accuracy threshold")
                continue
            
            # Limit features if specified
            if max_features_per_model:
                high_acc_features = high_acc_features[:max_features_per_model]
            
            print(f"  Found {len(high_acc_features)} high-accuracy features")
            
            # Load the SAE model
            sae_model = load_sae_model(model_num)
            
            # Convert feature numbers to indices
            feature_indices = []
            for feature_num, explanation, accuracy in high_acc_features:
                # Extract numeric index from feature_num (e.g., "62" from "feature_62")
                if feature_num.startswith("feature_"):
                    idx = int(feature_num.split("_")[1])
                else:
                    idx = int(feature_num)
                
                feature_indices.append(idx)
                
                # Store metadata
                metadata["feature_details"].append({
                    "model_num": model_num,
                    "feature_num": feature_num,
                    "feature_idx": idx,
                    "explanation": explanation,
                    "accuracy": accuracy
                })
            
            # Create probes for these features
            model_probes = create_feature_probes(sae_model, feature_indices)
            all_probes.extend(model_probes)
            
            # Update metadata
            metadata["models"][model_num] = {
                "num_features": len(high_acc_features),
                "features": [(fn, acc) for fn, _, acc in high_acc_features]
            }
            
            print(f"  Created {len(model_probes)} feature probes")
            
        except Exception as e:
            print(f"  Error processing model {model_num}: {e}")
            continue
    
    # Check if we have any probes
    if not all_probes:
        print("\nNo valid features found across all models!")
        return None
    
    # Create combined FeatureProbe model
    combined_probe = FeatureProbe(all_probes)
    metadata["total_features"] = len(all_probes)
    
    # Determine output path
    if output_path is None:
        model_str = "_".join(model_nums)
        output_path = f"./lanse_models/lanse_model_combined_{model_str}.pth"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    torch.save(combined_probe.state_dict(), output_path)
    print(f"\n✅ Saved LanSE model to: {output_path}")
    print(f"   Total features: {metadata['total_features']}")
    
    # Save metadata if requested
    if save_metadata:
        metadata_path = output_path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Metadata saved to: {metadata_path}")
    
    # Save in LanSE_labeling format
    lanse_dir = os.path.dirname(output_path)
    model_folder, explanation_folder = save_lanse_format_for_class(metadata, lanse_dir)
    
    # Also save the model in the expected location for LanSE_labeling
    model_save_path = os.path.join(model_folder, "cxr.pt")
    torch.save(combined_probe, model_save_path)  # Save entire model, not just state_dict
    print(f"   Saved CXR model to: {model_save_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("LANSE MODEL CREATION SUMMARY")
    print("=" * 50)
    for model_num, model_info in metadata["models"].items():
        print(f"Model {model_num}: {model_info['num_features']} features")
    print(f"Total features: {metadata['total_features']}")
    
    # Show top features
    if metadata["feature_details"]:
        print("\nTop 5 features by accuracy:")
        sorted_features = sorted(metadata["feature_details"], key=lambda x: x["accuracy"], reverse=True)[:5]
        for i, feat in enumerate(sorted_features, 1):
            print(f"{i}. Model {feat['model_num']}, Feature {feat['feature_num']}: "
                  f"{feat['explanation'][:50]}... ({feat['accuracy']:.1%})")
    
    return combined_probe

# =================== Utility Functions ===================

def save_lanse_explanations(feature_details: List[Dict], output_path: str):
    """Save feature explanations in LanSE format
    
    Args:
        feature_details: List of feature metadata dictionaries
        output_path: Path to save the explanations JSONL file
    """
    with open(output_path, 'w') as f:
        for idx, feature in enumerate(feature_details):
            # Create LanSE format entry for each feature
            lanse_entry = {
                "dimension_idx": idx,  # The index in the LanSE model
                "model_num": feature["model_num"],
                "original_feature_num": feature["feature_num"],
                "original_feature_idx": feature["feature_idx"],
                "explanation": feature["explanation"],
                "accuracy": feature["accuracy"]
            }
            f.write(json.dumps(lanse_entry) + '\n')

class LanSE_labeling(nn.Module):
    """LanSE model for chest X-ray analysis with interpretable features"""
    def __init__(self, model_folder: str, explanation_folder: str, device=None):
        super().__init__()
        self.groups = ["cxr"]  # Single group for chest X-ray
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load explanations and models
        self.explanations = {}
        self.joint_IntSE = {}
        
        for group in self.groups:
            # Load explanations
            explanation_path = os.path.join(explanation_folder, f"{group}.jsonl")
            self.explanations[group] = {}
            
            with open(explanation_path, 'r') as f:
                for line in f:
                    obj = json.loads(line.strip())
                    idx = obj["dimension_idx"]  # Using dimension_idx from our format
                    self.explanations[group][idx] = obj["explanation"]
            
            # Load model
            model_path = os.path.join(model_folder, f"{group}.pt")
            probe_model = torch.load(model_path, map_location=self.device, weights_only=False)
            probe_model.eval()
            self.joint_IntSE[group] = probe_model
        
        # Load BiomedCLIP for encoding
        print("Loading BiomedCLIP model...")
        from open_clip import create_model_from_pretrained, get_tokenizer
        self.biomedclip, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.biomedclip.to(self.device)
        self.biomedclip.eval()
        
        # Set thresholds for chest X-ray features
        self.threshold = {
            "cxr": 5  # Adjust based on your needs
        }
        
        print("-" * 100)
        print("The Medical LanSE model for chest X-ray analysis has been initialized.")
        for group in self.groups:
            print(f"Group: {group.upper()}, Number of features: {len(self.explanations[group])}, Threshold: {self.threshold[group]}")
        print("-" * 50)
    
    def _encode(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image and text using BiomedCLIP"""
        # Process image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Process text
        text_tokens = self.tokenizer([text], context_length=256).to(self.device)
        
        with torch.no_grad():
            image_feat = self.biomedclip.encode_image(image_tensor)
            text_feat = self.biomedclip.encode_text(text_tokens)
            
            # Normalize features
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        return image_feat, text_feat
    
    def _get_sparse_activations(self, image: Image.Image, text: str) -> Tuple[Dict, Dict]:
        """Get sparse activations for the input"""
        image_feat, text_feat = self._encode(image, text)
        
        z_sparse = {}
        z_mask = {}
        
        for group in self.groups:
            # Concatenate image and text features for CXR analysis
            combined_feat = torch.cat([image_feat, text_feat], dim=1)
            
            with torch.no_grad():
                z_sparse[group] = self.joint_IntSE[group](combined_feat)
                z_mask[group] = (z_sparse[group] > self.threshold[group]).squeeze(0)
        
        return z_sparse, z_mask
    
    def _explanations(self, image: Image.Image, text: str) -> Dict[str, List[Dict]]:
        """Get explanations for activated features"""
        z_sparse, z_mask = self._get_sparse_activations(image, text)
        
        explanations = {}
        for group in self.groups:
            indices = z_mask[group].nonzero(as_tuple=True)[0].tolist()
            explanations[group] = [
                {
                    'index': i, 
                    'explanation': self.explanations[group][i],
                    'activation': z_sparse[group][0, i].item()
                } 
                for i in indices if i in self.explanations[group]
            ]
            # Sort by activation strength
            explanations[group].sort(key=lambda x: x['activation'], reverse=True)
        
        return explanations
    
    def forward(self, image: Image.Image, text: str) -> torch.Tensor:
        """Forward pass returning sparse activations"""
        z_sparse, _ = self._get_sparse_activations(image, text)
        return z_sparse["cxr"]
    
    def get_activated_features(self, image: Image.Image, text: str, top_k: int = None) -> List[Dict]:
        """Get activated features with explanations, optionally limited to top_k"""
        explanations = self._explanations(image, text)
        features = explanations.get("cxr", [])
        
        if top_k and len(features) > top_k:
            features = features[:top_k]
        
        return features

def save_lanse_format_for_class(metadata: Dict, output_dir: str):
    """Save the model and explanations in the format expected by LanSE_labeling class"""
    
    # Create output directory structure
    model_folder = os.path.join(output_dir, "models")
    explanation_folder = os.path.join(output_dir, "explanations")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(explanation_folder, exist_ok=True)
    
    # Save explanations in the expected format
    explanation_path = os.path.join(explanation_folder, "cxr.jsonl")
    with open(explanation_path, 'w') as f:
        for idx, feature in enumerate(metadata["feature_details"]):
            entry = {
                "index": idx,
                "dimension_idx": idx,
                "explanation": feature["explanation"],
                "accuracy": feature["accuracy"],
                "model_num": feature["model_num"],
                "original_feature": feature["feature_num"]
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved CXR explanations to: {explanation_path}")
    
    # The model should be saved as cxr.pt in the models folder
    # This is handled by renaming/copying the main model file
    return model_folder, explanation_folder

def load_lanse_model(model_path: str, num_features: int = None) -> FeatureProbe:
    """Load a saved LanSE model
    
    Args:
        model_path: Path to the saved model
        num_features: Number of features (if known, for verification)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load metadata if available
    metadata_path = model_path.replace('.pth', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if num_features is None:
                num_features = metadata["total_features"]
    
    # Create empty probe list
    if num_features is None:
        # Try to infer from state dict
        state_dict = torch.load(model_path, map_location=device)
        num_features = len([k for k in state_dict.keys() if k.startswith("probes.")])
    
    # Create dummy probes with correct dimensions
    # Note: You'll need to know the input dimension
    input_dim = config.get("training", {}).get("input_dim", 1024)
    dummy_probes = [nn.Linear(input_dim, 1) for _ in range(num_features)]
    
    # Create model and load weights
    model = FeatureProbe(dummy_probes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# =================== Main Execution ===================

if __name__ == "__main__":
    # Configuration
    MODEL_NUMS = []
    for i in range(1, 99):
        MODEL_NUMS.append(str(i))
    ACCURACY_THRESHOLD = 0.99  # Minimum accuracy for feature inclusion
    MAX_FEATURES_PER_MODEL = None  # Set to limit features per model (None for all)
    
    # Create LanSE model from high-accuracy features
    lanse_model = create_lanse_model(
        model_nums=MODEL_NUMS,
        accuracy_threshold=ACCURACY_THRESHOLD,
        max_features_per_model=MAX_FEATURES_PER_MODEL,
        output_path="./lanse_models/medical_lanse_high_accuracy.pth",
        save_metadata=True
    )
    
    # Optional: Test loading the model
    if lanse_model is not None:
        print("\nTesting model loading...")
        
        # Test using the LanSE_labeling class
        print("\nTesting LanSE_labeling interface...")
        output_path = ""
        lanse_dir = os.path.dirname(output_path)
        model_folder = ""
        explanation_folder = ""
        
        # Initialize the LanSE_labeling model
        lanse = LanSE_labeling(model_folder, explanation_folder, device=device)
        
        print(f"\nSuccessfully loaded LanSE model for CXR analysis")
        print(f"Total features: {len(lanse.explanations['cxr'])}")
        
        # Example usage (uncomment with actual data):
        # from PIL import Image
        # image = Image.open("path/to/cxr_image.png")
        # text = "chest x-ray showing consolidation in the right lower lobe"
        # 
        # # Get activated features
        # features = lanse.get_activated_features(image, text, top_k=5)
        # print("\nTop activated features:")
        # for feat in features:
        #     print(f"  {feat['explanation']}: {feat['activation']:.3f}")