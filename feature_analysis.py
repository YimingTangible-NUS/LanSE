import os
import json
import random
import yaml
import torch
import io
import base64
from openai import OpenAI
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = config["general"]["cuda_devices"]

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

def extract_explanation(text: str) -> str:
    """Extract the key explanation from the model's response."""
    # Clean up the response to get the core explanation
    if "[" in text and "]" in text:
        # Extract content between brackets if present
        start = text.find("[")
        end = text.find("]", start)
        if start != -1 and end != -1:
            return text[start+1:end].strip()
    
    # Otherwise, try to extract the main finding
    # Look for key phrases that indicate the main finding
    key_phrases = [
        "common feature is",
        "commonality is", 
        "all images show",
        "shared characteristic is",
        "consistent finding is"
    ]
    
    text_lower = text.lower()
    for phrase in key_phrases:
        if phrase in text_lower:
            idx = text_lower.find(phrase)
            # Extract the sentence containing this phrase
            start = text.rfind('.', 0, idx) + 1
            end = text.find('.', idx)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
    
    # If no key phrases found, return first sentence after removing system messages
    sentences = text.split('.')
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # Skip very short sentences
            return sentence.strip()
    
    return text.strip()

def ensemble_feature_names(explanations: List[str]) -> str:
    """Ensemble multiple explanations to get the most consistent feature name."""
    if not explanations:
        return None
    
    # Count occurrences of key terms
    term_counts = {}
    
    for explanation in explanations:
        if explanation:
            # Convert to lowercase for comparison
            explanation_lower = explanation.lower()
            # Split into words and look for medical/anatomical terms
            words = explanation_lower.split()
            
            # Common medical terms to look for
            medical_terms = [
                'opacity', 'consolidation', 'effusion', 'pneumothorax',
                'cardiomegaly', 'atelectasis', 'edema', 'infiltrate',
                'nodule', 'mass', 'fracture', 'pneumonia', 'catheter',
                'tube', 'device', 'heart', 'lung', 'chest', 'rib',
                'mediastinum', 'pleural', 'bilateral', 'unilateral',
                'left', 'right', 'upper', 'lower', 'lobe'
            ]
            
            for term in medical_terms:
                if term in explanation_lower:
                    term_counts[term] = term_counts.get(term, 0) + 1
    
    # Return the most common term
    if term_counts:
        return max(term_counts.items(), key=lambda x: x[1])[0]
    
    # If no medical terms found, return the first explanation
    return explanations[0] if explanations else None

def sample_files(feature_path: str, num_samples: int) -> List[Dict[str, str]]:
    """Randomly sample num_samples image-text pairs from the feature folder."""
    feature_path = Path(feature_path)
    
    # Find all image files (we'll match them with text files)
    image_files = list(feature_path.glob("*.png")) + list(feature_path.glob("*.jpg"))
    
    # Filter to only those with matching text files
    valid_pairs = []
    for img_file in image_files:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists() and img_file.name != "activations.txt":
            valid_pairs.append((img_file, txt_file))
    
    # Sample from valid pairs
    if not valid_pairs:
        print(f"No valid image-text pairs found in {feature_path}")
        return []
    
    sampled_pairs = random.sample(valid_pairs, min(num_samples, len(valid_pairs)))
    
    samples = []
    for img_path, txt_path in sampled_pairs:
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                text_content = file.read().strip()
            samples.append({
                "text": text_content,
                "image": str(img_path),
                "basename": img_path.stem
            })
        except Exception as e:
            print(f"Error reading files {img_path.stem}: {e}")
            continue
    
    return samples

def generate_explanation(samples: List[Dict[str, str]], model: str = "anthropic/claude-3.7-sonnet") -> Optional[str]:
    """Use multimodal LLM to analyze commonalities in sampled features."""
    if not samples:
        return None
    
    # Create messages array with both text and images
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert radiologist analyzing chest X-ray images and their associated reports. "
                "Your task is to identify common radiological features or patterns that appear across multiple images. "
                "Focus on specific anatomical structures, pathological findings, or technical aspects of the images."
            )
        }
    ]
    
    # Prepare the combined message with all samples
    message_content = []
    
    # Add text description
    message_content.append({
        "type": "text",
        "text": "I'm showing you multiple chest X-ray images with their associated radiology reports. Please analyze them carefully.\n\n"
    })
    
    # Add each sample's text and image
    for i, sample in enumerate(samples):
        try:
            # Add the text report
            message_content.append({
                "type": "text",
                "text": f"Report {i+1}: {sample['text'][:500]}...\n"  # Truncate long texts
            })
            
            # Load and encode image
            image = Image.open(sample['image'])
            
            # Convert CMYK to RGB if necessary
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            elif image.mode == 'L':  # Grayscale
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            # Resize image to reduce size and API costs
            max_size = 512  # Reduced size for faster processing
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True, quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Add image to message
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
            
        except Exception as e:
            print(f"Error processing sample {i} ({sample.get('basename', 'unknown')}): {e}")
            continue
    
    # Add the analysis request
    message_content.append({
        "type": "text",
        "text": (
            "\n\nBased on these chest X-ray images and reports, identify the ONE most prominent common feature "
            "or pattern that appears across ALL or MOST of these samples. "
            "Be specific and concise. Focus on radiological findings, anatomical structures, or pathological patterns. "
            "Provide your answer in the format: [COMMON FEATURE], For example, [The existence of bilateral pleural effusions] or [The existence of atelectasis]."
        )
    })
    
    messages.append({"role": "user", "content": message_content})
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200,
            temperature=0.3  # Lower temperature for more consistent responses
        )
        
        if completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            response = completion.choices[0].message.content
            return extract_explanation(response)
        else:
            print(f"Unexpected API response structure")
            return None
            
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def feature_analysis(model_num: str):
    """Analyze features for a given model."""
    # Parameters
    SAMPLES_PER_ITERATION = config.get("analysis", {}).get("samples_per_iteration", 10)
    ITERATIONS = config.get("analysis", {}).get("iterations", 1)
    MIN_SAMPLES_REQUIRED = config.get("analysis", {}).get("min_samples", 3)
    
    # Updated base path for combined features
    base_path = f""
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        return
    
    # Output file
    output_dir = "./analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"combined_feature_analysis_model{model_num}.jsonl")
    
    # Get all feature directories
    feature_dirs = [d for d in os.listdir(base_path) if d.startswith("feature_") and os.path.isdir(os.path.join(base_path, d))]
    feature_dirs.sort(key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    
    print(f"Found {len(feature_dirs)} feature directories in {base_path}")
    
    # Process each feature
    for feature_dir in feature_dirs:
        feature_path = os.path.join(base_path, feature_dir)
        feature_num = feature_dir.split("_")[1] if "_" in feature_dir else feature_dir
        
        # Count valid image-text pairs
        image_files = [f for f in os.listdir(feature_path) if f.endswith((".png", ".jpg"))]
        valid_pairs = []
        for img_file in image_files:
            txt_file = img_file.rsplit(".", 1)[0] + ".txt"
            if txt_file in os.listdir(feature_path) and txt_file != "activations.txt":
                valid_pairs.append((img_file, txt_file))
        
        if len(valid_pairs) < MIN_SAMPLES_REQUIRED:
            print(f"Skipping Feature {feature_num} - only {len(valid_pairs)} valid pair(s)")
            continue
        
        print(f"Processing Feature {feature_num} with {len(valid_pairs)} pairs...")
        
        # Run multiple iterations if specified
        explanations = []
        for iteration in range(ITERATIONS):
            samples = sample_files(feature_path, SAMPLES_PER_ITERATION)
            if samples:
                explanation = generate_explanation(samples)
                if explanation:
                    explanations.append(explanation)
                    print(f"  Iteration {iteration + 1}: {explanation}")
        
        # Ensemble explanations if multiple iterations
        if len(explanations) > 1:
            final_explanation = ensemble_feature_names(explanations)
        elif explanations:
            final_explanation = explanations[0]
        else:
            final_explanation = "Unable to determine common feature"
        
        # Create record
        record = {
            "model_num": model_num,
            "feature_num": feature_num,
            "feature_path": feature_path,
            "num_samples": len(valid_pairs),
            "explanation": final_explanation,
            "all_explanations": explanations if ITERATIONS > 1 else None
        }
        
        # Save to file
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        print(f"Completed Feature {feature_num}: {final_explanation}\n")
    
    print(f"Analysis completed. Results saved to {output_file}")

def analyze_specific_feature(model_num: str, feature_num: str):
    """Analyze a specific feature for debugging."""
    feature_path = f""
    
    if not os.path.exists(feature_path):
        print(f"Feature path does not exist: {feature_path}")
        return
    
    print(f"Analyzing specific feature: {feature_path}")
    
    # Sample files and generate explanation
    samples = sample_files(feature_path, 10)
    if samples:
        print(f"Found {len(samples)} samples")
        explanation = generate_explanation(samples)
        print(f"Explanation: {explanation}")
        return explanation
    else:
        print("No valid samples found")
        return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_num = config['general']['model_num']
    
    for model_num in range(1, 100):
        model_num = str(model_num)
        print(f"Starting analysis for model {model_num}")
        # Run full analysis
        feature_analysis(model_num)
    
        # Optional: Analyze a specific feature for debugging
        # analyze_specific_feature("1", "62")