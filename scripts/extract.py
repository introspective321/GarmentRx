import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
_MODEL_CACHE = None

def load_resnet50(device='cpu'):
    """Load pre-trained ResNet50, remove final layer"""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        print("Using cached ResNet50")
        return _MODEL_CACHE
    
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model = model.to(device).eval()
    _MODEL_CACHE = model
    return model

def extract_vector(png_path, device='cpu'):
    """Extract 2048D feature vector from PNG"""
    img = Image.open(png_path).convert('RGBA')
    alpha = np.array(img)[:, :, 3]
    img_rgb = img.convert('RGB')
    
    # Mask non-clothing (alpha=0) as black
    img_array = np.array(img_rgb)
    img_array[alpha == 0] = [0, 0, 0]
    img = Image.fromarray(img_array)
    
    # Transform for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Extract vector
    model = load_resnet50(device)
    with torch.no_grad():
        vector = model(img_tensor).flatten().cpu().numpy()
    return vector.tolist()

def extract_color(png_path):
    """Extract dominant color using HSV"""
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]
    else:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Mask non-clothing
    mask = (alpha > 0) & (s > 0.2 * 255)  # Ignore low saturation
    hues = h[mask]
    if len(hues) == 0:
        return "neutral"
    
    # Histogram of hues
    hist, bins = np.histogram(hues, bins=36, range=(0, 180))
    dominant_hue = bins[np.argmax(hist)]
    
    # Map hue to color
    if 0 <= dominant_hue <= 15 or 165 <= dominant_hue <= 180:
        return "red"
    elif 16 <= dominant_hue <= 45:
        return "orange"
    elif 46 <= dominant_hue <= 75:
        return "yellow"
    elif 76 <= dominant_hue <= 105:
        return "green"
    elif 106 <= dominant_hue <= 135:
        return "blue"
    else:
        return "purple"

def extract_metadata(png_path):
    """Extract type and style using heuristics"""
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    brightness = np.mean(v[v > 0]) / 255.0 if np.any(v > 0) else 0.5
    
    # Type: Hardcoded from segment.py (class 3)
    type_ = "dress"
    
    # Style: Brightness-based
    style = "formal" if brightness < 0.7 else "casual"
    
    return type_, style

def process_and_save(png_path, output_dir='output', device='cpu'):
    """Process PNG, save features to JSON"""
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG not found: {png_path}")
    
    # Setup output dir
    features_dir = os.path.join(output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(png_path))[0]
    
    # Extract features
    vector = extract_vector(png_path, device)
    color = extract_color(png_path)
    type_, style = extract_metadata(png_path)
    
    # Save JSON
    features = {
        "vector": vector,
        "color": color,
        "type": type_,
        "style": style
    }
    json_path = os.path.join(features_dir, f"{filename}_features.json")
    with open(json_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Saved features: {json_path}")
    
    return json_path

def process_input_path(input_path, output_dir='output', device='cpu'):
    """Determine if input_path is a file or directory and process accordingly"""
    if os.path.isdir(input_path):
        json_paths = []
        for entry in os.listdir(input_path):
            file_path = os.path.join(input_path, entry)
            if os.path.isfile(file_path) and file_path.lower().endswith('.png'):
                try:
                    json_path = process_and_save(file_path, output_dir, device)
                    json_paths.append(json_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        return json_paths
    elif os.path.isfile(input_path):
        return [process_and_save(input_path, output_dir, device)]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--png_path', type=str, required=True, 
                        help='Path to a PNG image or directory of PNG images')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    try:
        result_paths = process_input_path(args.png_path, args.output_dir, args.device)
        if result_paths:
            for path in result_paths:
                print(f"Processing complete: {path}")
        else:
            print("No images found to process.")
    except Exception as e:
        print(f"Error: {str(e)}")