import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
import argparse
import glob
from tqdm import tqdm
from networks import U2NET
_MODEL_CACHE = {}

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # Remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model

class Normalize_image:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize_3 = transforms.Normalize([mean] * 3, [std] * 3)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        raise ValueError("Expected 3-channel tensor")

def apply_transform(img):
    transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
    return transforms.Compose(transforms_list)(img)

def load_seg_model(checkpoint_path, device='cpu'):
    global _MODEL_CACHE
    cache_key = f"{checkpoint_path}_{device}"
    
    if cache_key in _MODEL_CACHE:
        print("Using cached model")
        return _MODEL_CACHE[cache_key]
    
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    _MODEL_CACHE[cache_key] = net
    return net

def crop_top_region(img, crop_ratio=0.3):
    """
    Crop top portion of image to reduce face saliency
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    crop_h = int(h * crop_ratio)
    cropped = img_array[crop_h:, :, :]
    return Image.fromarray(cropped).convert("RGB")

def generate_mask(input_image, net, device='cpu', crop=False):
    """
    Args:
        input_image: PIL Image or path
        net: U2NET model
        device: 'cpu' or 'cuda'
        crop: Crop top 30% if True
        
    Returns:
        alpha_masks: Dict of class masks {cls: PIL Image}
    """
    if isinstance(input_image, str):
        img = Image.open(input_image).convert('RGB')
    else:
        img = input_image
    
    if crop:
        img = crop_top_region(img, crop_ratio=0.3)
    
    img_size = img.size
    img_resized = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img_resized)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    alpha_masks = {}
    for cls in range(1, 4):  # Classes 1-3 (skip background)
        if np.any(output_arr == cls):
            alpha_mask = (output_arr == cls).astype(np.uint8) * 255
            alpha_mask = alpha_mask[0]
            # Smooth mask
            alpha_mask = cv2.dilate(alpha_mask, np.ones((3, 3), np.uint8), iterations=1)
            alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
            alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
            alpha_masks[cls] = alpha_mask_img
    
    return alpha_masks

def extract_cloth(img, alpha_mask):
    """
    Args:
        img: PIL Image (RGB)
        alpha_mask: PIL Image (L)
        
    Returns:
        cloth_img: PIL Image (RGBA, transparent)
    """
    img_np = np.array(img)
    mask_np = np.array(alpha_mask)
    
    transparent = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    transparent[:, :, 0:3] = img_np
    transparent[:, :, 3] = mask_np
    
    return Image.fromarray(transparent, 'RGBA')

def process_image(image_path, output_dir, model, device, crop=False):
    """
    Args:
        image_path: Input image path
        output_dir: Output directory
        model: Pre-loaded model
        device: Device to use
        crop: Whether to crop top portion
        
    Returns:
        str: Path to extracted dress or None if failed
    """
    try:
        # Setup directories
        extracted_out_dir = os.path.join(output_dir, 'extracted')
        alpha_out_dir = os.path.join(output_dir, 'alpha')
        os.makedirs(extracted_out_dir, exist_ok=True)
        os.makedirs(alpha_out_dir, exist_ok=True)
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        img = Image.open(image_path).convert('RGB')
        
        alpha_masks = generate_mask(img, model, device, crop)
        
        dress_cls = 3
        if dress_cls not in alpha_masks:
            dress_cls = 2 if 2 in alpha_masks else (1 if 1 in alpha_masks else None)
            if dress_cls is None:
                print(f"Warning: No clothing detected in {image_path}")
                return None
        
        dress_mask = alpha_masks[dress_cls]
        
        # debugging
        alpha_path = os.path.join(alpha_out_dir, f"{filename}_mask_{dress_cls}.png")
        dress_mask.save(alpha_path)
        
        # extract and save dress
        dress_img = extract_cloth(img, dress_mask)
        dress_path = os.path.join(extracted_out_dir, f"{filename}_cloth.png")
        dress_img.save(dress_path)
        
        return dress_path
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_directory(input_dir, output_dir='output', model_path='models/cloth_segm.pth', crop=False, 
                      extensions=('jpg', 'jpeg', 'png')):
    """
    Process all images in a directory

    Returns:
        dict: Results with successful and failed files
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model once
    model = load_seg_model(model_path, device=device)
    
    # Get all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return {"successful": [], "failed": []}
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = []
    failed = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_image(image_path, output_dir, model, device, crop)
        if result:
            successful.append((image_path, result))
        else:
            failed.append(image_path)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)} images")
    print(f"Failed to process: {len(failed)} images")
    
    return {
        "successful": successful,
        "failed": failed
    }

def process_and_save(image_path, output_dir='output', model_path='models/cloth_segm.pth', crop=False):
    """
    Process a single image, extract dress (class 3), save PNG
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Setup directories
    extracted_out_dir = os.path.join(output_dir, 'extracted')
    alpha_out_dir = os.path.join(output_dir, 'alpha')
    os.makedirs(extracted_out_dir, exist_ok=True)
    os.makedirs(alpha_out_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_seg_model(model_path, device=device)
    
    img = Image.open(image_path).convert('RGB')
    
    alpha_masks = generate_mask(img, model, device, crop)
    
    dress_cls = 3
    if dress_cls not in alpha_masks:
        print(f"Warning: No dress (class 3) detected, trying class 2")
        dress_cls = 2 if 2 in alpha_masks else (1 if 1 in alpha_masks else None)
        if dress_cls is None:
            raise ValueError("No clothing classes (1-3) detected")
    
    dress_mask = alpha_masks[dress_cls]
    
    # debugging
    alpha_path = os.path.join(alpha_out_dir, f"{filename}_mask_{dress_cls}.png")
    dress_mask.save(alpha_path)
    print(f"Saved alpha mask: {alpha_path}")
    
    dress_img = extract_cloth(img, dress_mask)
    dress_path = os.path.join(extracted_out_dir, f"{filename}_cloth.png")
    dress_img.save(dress_path)
    print(f"Saved dress PNG: {dress_path}")
    
    return dress_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dress Segmentation')
    parser.add_argument('--image_path', type=str, help='Input image path')
    parser.add_argument('--input_dir', type=str, help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--model_path', type=str, default='models/cloth_segm.pth', 
                        help='Model checkpoint path')
    parser.add_argument('--crop', action='store_true', help='Crop top 30% to avoid faces')
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Process entire directory
        try:
            results = process_directory(
                args.input_dir, 
                args.output_dir, 
                args.model_path, 
                args.crop
            )
            print(f"Results saved to {args.output_dir}")
        except Exception as e:
            print(f"Error processing directory: {str(e)}")
    elif args.image_path:
        # Process single image
        try:
            dress_path = process_and_save(args.image_path, args.output_dir, args.model_path, args.crop)
            print(f"Processing complete: {dress_path}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        parser.error("Either --image_path or --input_dir must be provided")