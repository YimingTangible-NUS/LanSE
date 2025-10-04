import os
import json
import random
import time
import yaml
from openai import OpenAI
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def extract_answer(answer: str) -> int:
    """Extract yes/no answer from LMM response."""
    answer_lower = answer.lower()
    
    # Check for clear yes/no at the beginning
    if answer_lower.startswith("yes"):
        return 1
    elif answer_lower.startswith("no"):
        return 0
    
    # Check for yes/no anywhere in the response
    if "yes" in answer_lower and "no" not in answer_lower:
        return 1
    elif "no" in answer_lower and "yes" not in answer_lower:
        return 0
    
    # Check for affirmative/negative phrases
    affirmative_phrases = ["correct", "matches", "consistent", "accurate", "true", "agrees"]
    negative_phrases = ["incorrect", "does not match", "inconsistent", "inaccurate", "false", "disagrees"]
    
    has_affirmative = any(phrase in answer_lower for phrase in affirmative_phrases)
    has_negative = any(phrase in answer_lower for phrase in negative_phrases)
    
    if has_affirmative and not has_negative:
        return 1
    elif has_negative and not has_affirmative:
        return 0
    
    # Default to 0 if uncertain
    print(f"Warning: Could not clearly extract answer from: {answer[:100]}...")
    return 0

def load_feature_analysis(model_num: str) -> List[Dict]:
    """Load the feature analysis results for a given model."""
    analysis_file = f""
    
    if not os.path.exists(analysis_file):
        print(f"Analysis file not found: {analysis_file}")
        return []
    
    features = []
    with open(analysis_file, "r") as f:
        for line in f:
            try:
                feature = json.loads(line.strip())
                features.append(feature)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(features)} features from model {model_num}")
    return features

def load_image_text_pairs(feature_path: str, max_samples: int = 10) -> List[Tuple[str, str, str]]:
    """Load image-text pairs from a feature directory.
    
    Returns:
        List of tuples (image_path, text_content, basename)
    """
    feature_path = Path(feature_path)
    
    if not feature_path.exists():
        print(f"Feature path does not exist: {feature_path}")
        return []
    
    pairs = []
    
    # Find all image files
    image_files = list(feature_path.glob("*.png")) + list(feature_path.glob("*.jpg"))
    
    # Filter to valid pairs with text files
    for img_file in image_files:
        txt_file = img_file.with_suffix('.txt')
        
        # Skip activations.txt
        if txt_file.name == "activations.txt":
            continue
            
        if txt_file.exists():
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()
                pairs.append((str(img_file), text_content, img_file.stem))
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue
    
    # Randomly sample if we have more than max_samples
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    
    return pairs

def evaluate_single_sample(image_path: str, text_content: str, feature_explanation: str, 
                          model: str = "anthropic/claude-3-haiku") -> Tuple[str, int]:
    """Evaluate if a single sample matches the feature explanation."""
    
    try:
        # Load and prepare image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode == 'CMYK':
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        # Resize for efficiency
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare messages for the LMM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert radiologist evaluating chest X-ray images. "
                    "Your task is to determine if a given radiological finding or feature "
                    "is present in the image based on both the image and its radiology report."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Radiology Report:\n{text_content[:1000]}"  # Truncate long reports
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nFeature to verify: {feature_explanation}"
                    }
                ]
            },
            {
                "role": "user",
                "content": (
                    "Based on both the chest X-ray image and the radiology report above, "
                    "does this case demonstrate or contain the specified feature? "
                    "Consider both visual evidence in the image and textual evidence in the report. "
                    "Answer ONLY with 'yes' if the feature is clearly present, or 'no' if it is absent or unclear."
                )
            }
        ]
        
        # Make API call
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=0.1  # Low temperature for consistent yes/no answers
        )
        
        response = completion.choices[0].message.content
        label = extract_answer(response)
        
        return response, label
        
    except Exception as e:
        print(f"Error evaluating sample {image_path}: {e}")
        return f"Error: {str(e)}", 0

def evaluate_feature_accuracy(model_num: str, samples_per_feature: int = 10, 
                             max_features: Optional[int] = None,
                             output_dir: str = "./evaluation_results"):
    """Evaluate the accuracy of identified features for a model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"accuracy_evaluation_model{model_num}.jsonl")
    summary_file = os.path.join(output_dir, f"accuracy_summary_model{model_num}.json")
    
    # Load feature analysis results
    features = load_feature_analysis(model_num)
    
    if not features:
        print(f"No features found for model {model_num}")
        return
    
    # Limit features if specified
    if max_features:
        features = features[:max_features]
    
    print(f"Evaluating {len(features)} features with up to {samples_per_feature} samples each")
    
    # Track results
    all_results = []
    feature_accuracies = {}
    
    # Process each feature
    for feature_data in tqdm(features, desc="Evaluating features"):
        feature_num = feature_data["feature_num"]
        feature_path = feature_data["feature_path"]
        feature_explanation = feature_data["explanation"]
        
        print(f"\nEvaluating Feature {feature_num}: {feature_explanation}")
        
        # Load samples for this feature
        samples = load_image_text_pairs(feature_path, max_samples=samples_per_feature)
        
        if not samples:
            print(f"No valid samples found for feature {feature_num}")
            continue
        
        # Evaluate each sample
        feature_results = []
        correct_count = 0
        
        for img_path, text_content, basename in samples:
            response, label = evaluate_single_sample(img_path, text_content, feature_explanation)
            
            result = {
                "model_num": model_num,
                "feature_num": feature_num,
                "feature_explanation": feature_explanation,
                "image_path": img_path,
                "basename": basename,
                "text_snippet": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "llm_response": response,
                "llm_label": label
            }
            
            feature_results.append(result)
            correct_count += label
            
            # Save result immediately
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate feature accuracy
        if feature_results:
            accuracy = correct_count / len(feature_results)
            feature_accuracies[feature_num] = {
                "explanation": feature_explanation,
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(feature_results)
            }
            
            print(f"Feature {feature_num} accuracy: {accuracy:.2%} ({correct_count}/{len(feature_results)})")
            all_results.extend(feature_results)
    
    # Calculate overall statistics
    if all_results:
        overall_accuracy = sum(r["llm_label"] for r in all_results) / len(all_results)
        
        # Group by feature quality
        high_accuracy_features = {k: v for k, v in feature_accuracies.items() if v["accuracy"] >= 0.7}
        medium_accuracy_features = {k: v for k, v in feature_accuracies.items() if 0.3 <= v["accuracy"] < 0.7}
        low_accuracy_features = {k: v for k, v in feature_accuracies.items() if v["accuracy"] < 0.3}
        
        summary = {
            "model_num": model_num,
            "total_features_evaluated": len(feature_accuracies),
            "total_samples_evaluated": len(all_results),
            "overall_accuracy": overall_accuracy,
            "feature_accuracies": feature_accuracies,
            "high_accuracy_features": high_accuracy_features,
            "medium_accuracy_features": medium_accuracy_features,
            "low_accuracy_features": low_accuracy_features,
            "accuracy_distribution": {
                "high": len(high_accuracy_features),
                "medium": len(medium_accuracy_features),
                "low": len(low_accuracy_features)
            }
        }
        
        # Save summary
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print(f"EVALUATION SUMMARY FOR MODEL {model_num}")
        print("="*50)
        print(f"Total features evaluated: {len(feature_accuracies)}")
        print(f"Total samples evaluated: {len(all_results)}")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"\nAccuracy distribution:")
        print(f"  High accuracy (≥70%): {len(high_accuracy_features)} features")
        print(f"  Medium accuracy (30-70%): {len(medium_accuracy_features)} features")
        print(f"  Low accuracy (<30%): {len(low_accuracy_features)} features")
        
        if high_accuracy_features:
            print(f"\nTop performing features:")
            sorted_features = sorted(high_accuracy_features.items(), 
                                    key=lambda x: x[1]["accuracy"], 
                                    reverse=True)[:5]
            for feat_num, feat_data in sorted_features:
                print(f"  Feature {feat_num}: {feat_data['explanation'][:50]}... "
                      f"({feat_data['accuracy']:.1%})")
    
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")

def batch_evaluate_models(model_nums: List[str], **kwargs):
    """Evaluate multiple models in sequence."""
    for model_num in model_nums:
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL {model_num}")
        print('='*60)
        evaluate_feature_accuracy(model_num, **kwargs)
        print(f"\nCompleted evaluation for model {model_num}")
        time.sleep(2)  # Pause between models

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Single model evaluation
    # model_num = config['general'].get('model_num', '1')
    
    # Evaluation parameters
    samples_per_feature = 20  # Number of samples to evaluate per feature
    max_features = None  # Set to limit number of features (None for all)
    

    for model_num in range(1,99):
        # Run evaluation
        evaluate_feature_accuracy(
            model_num=model_num,
            samples_per_feature=samples_per_feature,
            max_features=max_features,
            output_dir="./evaluation_results"
        )
        
    # Optional: Evaluate multiple models
    # model_nums = ['1', '2', '3']
    # batch_evaluate_models(
    #     model_nums=model_nums,
    #     samples_per_feature=10,
    #     max_features=50
    # )