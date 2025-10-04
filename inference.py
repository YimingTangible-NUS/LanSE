import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple
from pathlib import Path

# Sparse Encoder Probes Wrapper
class FeatureProbe(nn.Module):
    def __init__(self, probes):
        super().__init__()
        self.probes = nn.ModuleList(probes)

    def forward(self, x):
        return torch.cat([probe(x) for probe in self.probes], dim=1)

# Trust the FeatureProbe Class
torch.serialization.add_safe_globals([FeatureProbe])

# Medical LanSE Model for CXR
class MedicalLanSE(nn.Module):
    def __init__(self, model_folder, explanation_folder, device=None):
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
            
            if not os.path.exists(explanation_path):
                raise FileNotFoundError(f"Explanation file not found: {explanation_path}")
            
            with open(explanation_path, 'r') as f:
                for line in f:
                    obj = json.loads(line.strip())
                    idx = int(obj["index"])
                    self.explanations[group][idx] = obj["explanation"]
            
            # Load model
            model_path = os.path.join(model_folder, f"{group}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            probe_model = torch.load(model_path, map_location=self.device, weights_only=False)
            probe_model.eval()
            self.joint_IntSE[group] = probe_model
        
        # Load BiomedCLIP for encoding
        print("Loading BiomedCLIP model for medical image encoding...")
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        self.biomedclip, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.biomedclip.to(self.device)
        self.biomedclip.eval()
        
        # Set thresholds for chest X-ray features
        self.threshold = {
            "cxr": 0.5  # Adjust based on validation results
        }
        
        print("-" * 100)
        print("Medical LanSE for Chest X-Ray Analysis Initialized")
        for group in self.groups:
            print(f"Group: {group.upper()}")
            print(f"  - Number of interpretable features: {len(self.explanations[group])}")
            print(f"  - Activation threshold: {self.threshold[group]}")
        print("-" * 100)
    
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
            
            group_explanations = []
            for i in indices:
                if i in self.explanations[group]:
                    group_explanations.append({
                        'index': i,
                        'explanation': self.explanations[group][i],
                        'activation': z_sparse[group][0, i].item()
                    })
            
            # Sort by activation strength
            group_explanations.sort(key=lambda x: x['activation'], reverse=True)
            explanations[group] = group_explanations
        
        return explanations
    
    def analyze_sample(self, image: Image.Image, text: str, save_results: bool = False, 
                       output_prefix: str = "sample") -> Dict:
        """Comprehensive analysis of a CXR sample"""
        
        # Get explanations
        explanations = self._explanations(image, text)
        
        # Get raw activations for statistics
        z_sparse, z_mask = self._get_sparse_activations(image, text)
        
        # Compile results
        results = {
            "input_text": text[:500] + "..." if len(text) > 500 else text,
            "groups": {}
        }
        
        for group in self.groups:
            activated_features = explanations[group]
            all_activations = z_sparse[group].squeeze().cpu().numpy()
            
            results["groups"][group] = {
                "num_activated": len(activated_features),
                "total_features": len(self.explanations[group]),
                "activation_rate": len(activated_features) / len(self.explanations[group]) * 100,
                "max_activation": float(all_activations.max()),
                "mean_activation": float(all_activations.mean()),
                "features": activated_features[:10]  # Top 10 features
            }
        
        # Save results if requested
        if save_results:
            # Save image
            image.save(f"{output_prefix}_image.png")
            
            # Save text
            with open(f"{output_prefix}_report.txt", "w") as f:
                f.write(text)
            
            # Save analysis results
            with open(f"{output_prefix}_analysis.json", "w") as f:
                json.dump(results, f, indent=2)
        
        return results


def test_single_sample(model: MedicalLanSE, image_path: str, text_path: str = None):
    """Test the model on a single CXR image-text pair"""
    
    print(f"\n{'='*80}")
    print(f"Testing sample: {image_path}")
    print('='*80)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    
    # Load or generate text
    if text_path is None:
        # Try to find corresponding text file
        text_path = image_path.replace(".png", ".txt").replace(".jpg", ".txt")
    
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            text = f.read().strip()
    else:
        print(f"Warning: Text file not found at {text_path}, using default text")
        text = "Chest X-ray image showing lung fields"
    
    # Analyze the sample
    results = model.analyze_sample(image, text, save_results=True, output_prefix="test_sample")
    
    # Print detailed results
    print(f"\nRadiology Report (first 200 chars):")
    print(f"  {text[:200]}...")
    
    for group in model.groups:
        group_results = results["groups"][group]
        print(f"\n{group.upper()} Analysis:")
        print(f"  Activated features: {group_results['num_activated']}/{group_results['total_features']} "
              f"({group_results['activation_rate']:.1f}%)")
        print(f"  Max activation: {group_results['max_activation']:.3f}")
        print(f"  Mean activation: {group_results['mean_activation']:.3f}")
        
        if group_results['features']:
            print(f"\n  Top activated features:")
            for i, feat in enumerate(group_results['features'][:5], 1):
                print(f"    {i}. Feature {feat['index']} (activation={feat['activation']:.3f}):")
                print(f"       {feat['explanation']}")


def test_batch_samples(model: MedicalLanSE, data_folder: str, num_samples: int = 5):
    """Test the model on multiple CXR samples from a folder"""
    
    print(f"\n{'='*80}")
    print(f"Batch Testing: {data_folder}")
    print('='*80)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(data_folder).glob(ext))
    
    if not image_files:
        print(f"No image files found in {data_folder}")
        return
    
    # Limit to num_samples
    import random
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    print(f"Testing {len(image_files)} samples...")
    
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n--- Sample {idx}/{len(image_files)}: {image_path.name} ---")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Try to find corresponding text
        text_path = image_path.with_suffix('.txt')
        if text_path.exists():
            with open(text_path, "r") as f:
                text = f.read().strip()
        else:
            text = f"Chest X-ray image {image_path.name}"
        
        # Get explanations
        explanations = model._explanations(image, text)
        
        # Store results
        sample_result = {
            "image": str(image_path),
            "num_activated": len(explanations["cxr"]),
            "top_features": explanations["cxr"][:3] if explanations["cxr"] else []
        }
        all_results.append(sample_result)
        
        # Print summary
        print(f"  Activated features: {len(explanations['cxr'])}")
        if explanations["cxr"]:
            print(f"  Top feature: {explanations['cxr'][0]['explanation']}")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print('='*80)
    
    avg_activated = sum(r["num_activated"] for r in all_results) / len(all_results)
    print(f"Average activated features: {avg_activated:.1f}")
    
    # Find most common features
    feature_counts = {}
    for result in all_results:
        for feat in result["top_features"]:
            exp = feat["explanation"]
            feature_counts[exp] = feature_counts.get(exp, 0) + 1
    
    if feature_counts:
        print("\nMost common features across samples:")
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for feat, count in sorted_features[:5]:
            print(f"  - {feat}: {count} samples")


if __name__ == "__main__":
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    # Load the Medical LanSE model
    explanation_folder = "./lanse_models/explanations"
    model_folder = "./lanse_models/models"
    
    print("Initializing Medical LanSE model...")
    model = MedicalLanSE(model_folder, explanation_folder)
    
    # Test on different types of samples
    
    # === Test 1: Single real MIMIC-CXR sample ===
    real_cxr_path = ""
    real_text_path = ""
    
    if os.path.exists(real_cxr_path):
        print("\n" + "="*80)
        print("TEST 1: Real MIMIC-CXR Sample")
        print("="*80)
        test_single_sample(model, real_cxr_path, real_text_path)
    
    # === Test 2: Single generated sample ===
    # generated_cxr_path = "/data0/yiming_tangible/generated_datasets/stable-diffusion-chest-xray-mimicCXR/generated_images/generated_2.png"
    # generated_text_path = "/data0/yiming_tangible/generated_datasets/stable-diffusion-chest-xray-mimicCXR/generated_images/generated_2.txt"
    
    # if os.path.exists(generated_cxr_path):
    #     print("\n" + "="*80)
    #     print("TEST 2: Generated CXR Sample")
    #     print("="*80)
    #     test_single_sample(model, generated_cxr_path, generated_text_path)
    
    # # === Test 3: Batch testing on multiple samples ===
    # batch_folder = "/data0/yiming_tangible/datasets/mimic-cxr-dataset/data"
    
    # if os.path.exists(batch_folder):
    #     print("\n" + "="*80)
    #     print("TEST 3: Batch Testing")
    #     print("="*80)
    #     test_batch_samples(model, batch_folder, num_samples=5)
    
    # # === Test 4: Compare real vs generated ===
    # print("\n" + "="*80)
    # print("TEST 4: Comparative Analysis")
    # print("="*80)
    
    # You can add specific comparison logic here
    # print("Testing complete. Results saved to test_sample_* files.")